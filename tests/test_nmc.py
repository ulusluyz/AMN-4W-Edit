"""CPU verification of AMN-NMC causal language model, gradient flow and data split."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from amn_nmc import AMNNMC, NMCConfig
from amn_nmc.corpus import prepare, load_split
from amn_nmc.generate import generate
from amn_nmc.train import chunks, evaluate, train


def tiny(**kwargs):
    return AMNNMC(NMCConfig(d_model=24, memory_dim=12, slots=4, layers=1, **kwargs))


def test_model_shape_real_byte_tokens():
    torch.manual_seed(17)
    model = tiny()
    x = torch.tensor([[0, 1, 255, 128, 195, 167], [33, 42, 127, 129, 0, 4]])
    logits, state = model(x)
    assert logits.shape == (2, 6, 256)
    assert torch.isfinite(logits).all()
    assert state.blocks[0].fast.shape == (2, 12, 12)
    assert state.blocks[0].keys.shape == (2, 4, 12)
    assert 10000 < sum(p.numel() for p in model.parameters()) < 40000


def test_full_sequence_same_as_state_carry_chunks():
    model = tiny().eval()
    x = torch.randint(0, 256, (2, 17), generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        full, full_state = model(x)
        first, state = model(x[:, :5])
        second, state = model(x[:, 5:9], state)
        third, state = model(x[:, 9:], state)
    assert torch.allclose(full, torch.cat([first, second, third], dim=1), atol=2e-5, rtol=2e-5)
    assert torch.allclose(full_state.blocks[0].fast, state.blocks[0].fast, atol=2e-5)
    assert torch.allclose(full_state.blocks[0].values, state.blocks[0].values, atol=2e-5)


def test_causality_future_tokens_do_not_change_past_logits():
    model = tiny().eval()
    a = torch.tensor([[1, 55, 91, 12, 230, 1]], dtype=torch.long)
    b = torch.tensor([[1, 55, 91, 12, 6, 250]], dtype=torch.long)
    with torch.no_grad():
        ya, _ = model(a)
        yb, _ = model(b)
    assert torch.allclose(ya[:, :4], yb[:, :4], atol=1e-6)


def test_memory_components_receive_language_loss_gradient():
    torch.manual_seed(33)
    model = tiny()
    tokens = torch.randint(0, 256, (2, 16))
    logits, _ = model(tokens[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 256), tokens[:, 1:].reshape(-1))
    loss.backward()
    names = [
        'blocks.0.gate.weight',
        'blocks.0.fast_write.weight',
        'blocks.0.fast_forget.weight',
        'blocks.0.long_write.weight',
        'blocks.0.long_allocate.weight',
        'blocks.0.key.weight',
        'blocks.0.slot_anchors',
        'blocks.0.read_merge.weight',
        'embedding.weight',
    ]
    params = dict(model.named_parameters())
    for name in names:
        g = params[name].grad
        assert g is not None, f'gradient disconnected: {name}'
        assert torch.isfinite(g).all(), f'nonfinite gradient: {name}'
        assert g.abs().sum().item() > 1e-12, f'zero gradient: {name}'


def test_memory_ablation_changes_logits_and_keeps_disabled_states_unchanged():
    x = torch.randint(0, 256, (1, 18), generator=torch.Generator().manual_seed(3))
    full = tiny()
    ablated = tiny(use_fast=False, use_episodic=False)
    ablated.load_state_dict(full.state_dict())
    with torch.no_grad():
        out, full_state = full(x)
        altered, off_state = ablated(x)
    assert not torch.allclose(out, altered)
    assert full_state.blocks[0].fast.abs().sum() > 0
    assert full_state.blocks[0].values.abs().sum() > 0
    assert torch.count_nonzero(off_state.blocks[0].fast) == 0
    assert torch.count_nonzero(off_state.blocks[0].values) == 0


def test_state_detach_and_variable_document_lengths():
    model = tiny()
    seq = 'Türkçe bellek.'.encode('utf-8')
    state = None
    for x,y in chunks(seq, 3):
        logits, state = model(x, state)
        loss = F.cross_entropy(logits.reshape(-1,256), y.reshape(-1))
        loss.backward()
        state = state.detach()
        assert all(t.grad_fn is None for b in state.blocks for t in (b.hidden,b.fast,b.keys,b.values,b.usage,b.read))
        model.zero_grad(set_to_none=True)
    result = evaluate(model, [seq, 'Merhaba dünya.'.encode('utf-8')], 4)
    assert math.isfinite(result['bpb']) and result['bytes'] > 0


def test_document_dedup_split_no_overlap(tmp_path):
    source = tmp_path/'raw'; source.mkdir()
    for i in range(15):
        (source/f'{i:02}.txt').write_text(f'{i}. Türkçe dil öğrenme için belge numarası {i}.',encoding='utf-8')
    # Identical text in another location must be deduplicated.
    (source/'duplicate.txt').write_text('0. Türkçe dil öğrenme için belge numarası 0.',encoding='utf-8')
    dest = tmp_path/'splits'
    manifest = prepare(source, dest, license_id='TEST_ONLY')
    assert sum(manifest['counts'].values()) == 15
    split_hashes = {split: {r['sha256'] for r in manifest['documents'] if r['split']==split}
                    for split in ('train','validation','test')}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert split_hashes[a].isdisjoint(split_hashes[b])
    assert all(load_split(dest,s) for s in split_hashes)
    assert manifest['license'] == 'TEST_ONLY'


def test_cpu_training_checkpoint_and_generation(tmp_path):
    raw = tmp_path/'raw';raw.mkdir()
    for i in range(18):
        (raw/f'{i}.txt').write_text(f'Türkçe bir dil modeli öğreniyor. Girdi {i}. Bellek ile dil birlikte çalışıyor.',encoding='utf-8')
    data = tmp_path/'data';prepare(raw,data,license_id='TEST_ONLY')
    conf = { 'model': {'d_model':16, 'memory_dim':8, 'slots':3, 'layers':1},
             'training': {'device':'cpu','threads':2,'lr':.001,'chunk_size':8,'epochs':1,'max_steps':3}}
    config_file = tmp_path/'config.json';config_file.write_text(json.dumps(conf))
    run = tmp_path/'run'
    result = train(data, run, config_file, seed=17)
    assert result['steps'] == 3
    assert math.isfinite(result['final_validation']['bpb'])
    assert (run/'initial.pt').exists() and (run/'last.pt').exists()
    assert (run/result['best_checkpoint']).exists()
    payload = torch.load(run/result['best_checkpoint'],map_location='cpu',weights_only=True)
    restored = AMNNMC(NMCConfig(**payload['config']))
    restored.load_state_dict(payload['model'])
    output1 = generate(restored, 'Merhaba', new_bytes=5, seed=42)
    output2 = generate(restored, 'Merhaba', new_bytes=5, seed=42)
    assert output1 == output2 and output1.startswith('Merhaba')
    assert (run/'training.jsonl').read_text().count('\n') == 3


def test_reject_invalid_token_shapes_and_out_of_range():
    model=tiny()
    with pytest.raises(ValueError):model(torch.tensor([0,1]))
    with pytest.raises(ValueError):model(torch.tensor([[256]]))
    with pytest.raises(ValueError):model(torch.empty((1,0),dtype=torch.long))
    with pytest.raises(ValueError):model(torch.tensor([[-1]]))
