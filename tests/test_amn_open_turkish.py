"""Pure AMN byte language model: causal, streaming, UTF-8, source split tests."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from amn_open_turkish import (
    AMNOpenTurkish,BOS,EOS,PAD,IGNORE,make_training_example,collate,
    make_prefix,save_checkpoint,load_checkpoint
)
from prepare_open_turkish import split_rows


def test_unseen_unicode_is_supported_without_vocabulary_training():
    text="Özgün bir cümle: İstanbul, Arğavi, ğ ş ı İ; 日本語 🤖"
    prefix=make_prefix(text)
    assert prefix.decode('utf-8').find(text)>0
    assert any(b>=128 for b in prefix)
    x,y=make_training_example(text,"Dilin kendine özgü sesleri ve yazımı vardır.",600)
    assert len(x)==len(y)
    assert y[0]==IGNORE
    assert y[len(prefix)-1]==IGNORE
    assert y[len(prefix)]!=IGNORE
    assert y[-1]==EOS


def test_causal_prefix_does_not_see_future_characters():
    torch.manual_seed(2)
    m=AMNOpenTurkish(width=32,key_dim=8,layers=1)
    m.eval()
    tokens=torch.tensor([[BOS,ord('a'),ord('b'),ord('c'),ord('d')]])
    changed=tokens.clone();changed[0,-1]=ord('z')
    with torch.no_grad():
        a=m(tokens);b=m(changed)
    assert torch.allclose(a[:,:-1],b[:,:-1],atol=1e-6,rtol=1e-5)


def test_streaming_equals_parallel():
    torch.manual_seed(3)
    m=AMNOpenTurkish(width=32,key_dim=8,layers=1)
    m.eval()
    tokens=torch.randint(1,254,(3,11))
    tokens[:,0]=BOS
    with torch.no_grad():
        full=m(tokens)
        state=None;parts=[]
        for t in range(tokens.size(1)):
            out,state=m.step(tokens[:,t],state)
            parts.append(out)
        incremental=torch.stack(parts,1)
    assert torch.allclose(full,incremental,atol=2e-5,rtol=2e-5),float((full-incremental).abs().max())


def test_forward_backward_and_checkpoint(tmp_path):
    torch.manual_seed(4)
    m=AMNOpenTurkish(width=32,key_dim=8,layers=1)
    rows=[make_training_example('Öğrenme nasıl olur?',
        'Düzenli uygulama bilgi edinmeyi kolaylaştırır.',256),
          make_training_example('Bugün hava nasıl?',
        'Hava durumu yerden yere değişebilir.',256)]
    x,y=collate(rows)
    loss=F.cross_entropy(m(x).reshape(-1,259),y.reshape(-1),ignore_index=IGNORE)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    saved=tmp_path/'ck.pt';save_checkpoint(saved,m,extra={'mock':True})
    copied,meta=load_checkpoint(saved)
    assert meta['mock']
    assert torch.equal(m(x),copied(x))
    reply=copied.complete('Bambaşka bir soru?',max_new_bytes=30,temperature=0,seed=8)
    assert isinstance(reply['text'],str)


def test_prompt_deduplicated_and_disjoint_data(tmp_path):
    import json
    samples=[]
    for i in range(45):
        samples.append({'instruction':f'Soru {i}: Bir bilgi hakkında açıklar mısın?',
                        'input':'','output':f'Bu {i} numaralı örnek açıklaması farklı ve öğrenilebilir bir metindir.'})
    samples.append(samples[0])
    path=tmp_path/'source.json'
    path.write_text(json.dumps(samples,ensure_ascii=False),encoding='utf-8')
    parts,stats=split_rows(path,384,100,10)
    def names(split):return {x['prompt'] for x in parts[split]}
    assert not names('train')&names('val')
    assert not names('train')&names('test')
    assert not names('val')&names('test')
    assert stats['duplicate']==0 if 'duplicate' in stats else stats['skipped']['duplicate']==1
    assert stats['selected']==45
