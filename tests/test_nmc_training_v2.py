"""Regression checks: exact TBPTT checkpoint resume, causal gradient, budgeted BPB."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import pytest
from torch.nn import functional as F

from amn_nmc import AMNNMC, NMCConfig
from amn_nmc.baseline_gru import GRUByteLM
from amn_nmc.train import train, evaluate, save_state, load_state, chunks


def toy_data(tmp_path):
    root = tmp_path / 'data'; root.mkdir()
    docs = [f'Türkçe nöral dil modeli bellek ve öğrenme. Deneme {i}. ' * 5 for i in range(7)]
    for name, texts in [('train', docs[:5]), ('validation', docs[5:6]), ('test', docs[6:])]:
        (root / (name + '.jsonl')).write_text(''.join(json.dumps({'text': t},ensure_ascii=False)+'\n' for t in texts))
    return root


def cfg(tmp_path, max_steps):
    p=tmp_path/f'cfg_{max_steps}.json'
    p.write_text(json.dumps({'model': {'d_model':16,'memory_dim':8,'slots':3,'layers':1},
                             'training':{'device':'cpu','threads':2,'lr':0.001,'chunk_size':12,
                                         'epochs':2,'max_steps':max_steps,'eval_every':3,
                                         'eval_max_bytes':100}}))
    return p


def test_resumed_training_is_bit_identical_to_uninterrupted(tmp_path):
    data=toy_data(tmp_path)
    a=train(data,tmp_path/'full',cfg(tmp_path,10),17)
    train(data,tmp_path/'partial',cfg(tmp_path,4),17)
    b=train(data,tmp_path/'partial',cfg(tmp_path,4),17,resume=True,extra_steps=6)
    assert (a['steps'],a['training_bytes'])==(b['steps'],b['training_bytes'])
    full=torch.load(tmp_path/'full'/'last.pt',map_location='cpu',weights_only=True)
    resumed=torch.load(tmp_path/'partial'/'last.pt',map_location='cpu',weights_only=True)
    assert full['cursor']==resumed['cursor']
    for key in full['model']:
        assert torch.equal(full['model'][key],resumed['model'][key]),key
    assert full['history']==resumed['history']
    assert len(full['recurrent_state'])==1


def test_training_rejects_modified_corpus_on_resume(tmp_path):
    root=toy_data(tmp_path)
    cfgfile=cfg(tmp_path,2)
    train(root,tmp_path/'run',cfgfile,17)
    with (root/'train.jsonl').open('a') as fp: fp.write('{"text":"extra facts"}\n')
    with pytest.raises(ValueError, match='Changed corpus'):
        train(root,tmp_path/'run',cfgfile,17,resume=True,extra_steps=2)


def test_budgeted_validation_never_claims_full_document_evaluation(tmp_path):
    data=toy_data(tmp_path)
    from amn_nmc.corpus import load_split
    docs=load_split(data,'validation')
    model=AMNNMC(NMCConfig(d_model=16,memory_dim=8,slots=3,layers=1))
    short=evaluate(model,docs,16,max_bytes=19)
    full=evaluate(model,docs,16)
    assert short['bytes']==19
    assert full['bytes']==len(docs[0])-1
    assert full['bytes']>short['bytes']


def test_state_pickle_contains_only_tensors_and_recovers(tmp_path):
    model=AMNNMC(NMCConfig(d_model=16,memory_dim=8,slots=3,layers=1))
    logits,s=model(torch.tensor([[1,2,3,4,5]],dtype=torch.long))
    data=save_state(s.detach());p=tmp_path/'st.pt';torch.save(data,p)
    recovered=load_state(torch.load(p,weights_only=True))
    with torch.no_grad():
        next1,_=model(torch.tensor([[6,7]]),s.detach())
        next2,_=model(torch.tensor([[6,7]]),recovered)
    assert torch.allclose(next1,next2)


def test_memory_matrix_and_slots_stay_finite_on_long_context():
    torch.manual_seed(5)
    model=AMNNMC(NMCConfig(d_model=20,memory_dim=10,slots=4,layers=1))
    torch.set_num_threads(2)
    h=None
    with torch.no_grad():
        for _ in range(20):
            x=torch.randint(0,256,(1,24))
            y,h=model(x,h)
            assert torch.isfinite(y).all()
            assert torch.isfinite(h.blocks[0].fast).all()
            assert torch.isfinite(h.blocks[0].values).all()
            assert 0 <= h.blocks[0].usage.min() and h.blocks[0].usage.max() <= 1.001


def test_stepped_updates_receive_gradient_not_label_lookup():
    torch.manual_seed(7)
    model=AMNNMC(NMCConfig(d_model=16,memory_dim=8,slots=3,layers=1))
    x=torch.randint(0,256,(2,9))
    y,_=model(x[:,:-1])
    loss=F.cross_entropy(y.reshape(-1,256),x[:,1:].reshape(-1))
    loss.backward()
    gradients=dict(model.named_parameters())
    for name in ('blocks.0.fast_write.weight','blocks.0.fast_forget.weight',
                 'blocks.0.long_write.weight','blocks.0.long_allocate.weight',
                 'blocks.0.slot_anchors'):
        grad=gradients[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name


def test_reference_gru_is_near_parameter_matched_but_independent():
    amn=AMNNMC(NMCConfig(d_model=48,memory_dim=24,slots=8,layers=2))
    gru=GRUByteLM(68,2)
    n=sum(p.numel() for p in amn.parameters())
    g=sum(p.numel() for p in gru.parameters())
    assert abs(n-g)/n<0.03
    assert all('gru' not in name for name,_ in amn.named_modules())
