"""Subword AMN: true autoregressive generation and end-to-end shape tests."""
import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT.parent/'amn_open_text'/'experiments'))
from amn_open_turkish_sp import AMNSubwordLM,encode,collate,save,restore,load_sp
from train_open_turkish_sp import build_tokenizer


def test_subword_turkish_utf8_and_training(tmp_path):
    train=[{'prompt':f'{word} nedir, nasıl açıklanır?',
            'answer':f'{word} öğrenme, iletişim ve düşünme süreçlerini geliştiren bir kavramdır.'}
           for word in ['öğrenme','çözüm','mühendislik','kalem','çevre','düşünce','başarı','dünya','algılama',
                        'bilgisayar','gelecek','enerji','bilgi','yaşam','arkadaş','insan','konuşma','okuma',
                        'yazma','teknoloji','programlama','gözlem','iletişim','kuram','bilim','öğretim']]
    sp=build_tokenizer(tmp_path,train,450)
    assert sp.vocab_size()>260
    text='Türkçe karakterleri ve bilinmeyen sözcüğü: çığışİğ 🤖'
    assert sp.decode(sp.encode(text))==text
    examples=[encode(sp,x['prompt'],x['answer'],192) for x in train[:8]]
    assert all(x for x in examples)
    tensor, gold=collate(examples,sp.pad_id())
    torch.manual_seed(0)
    m=AMNSubwordLM(sp.vocab_size(),width=32,key_dim=8,layers=1,pad_id=sp.pad_id())
    output=m(tensor)
    loss=F.cross_entropy(output.reshape(-1,sp.vocab_size()),gold.reshape(-1),ignore_index=-100)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    m.eval()
    with torch.no_grad():
        logits=m(tensor[:2,:14])
        state=None; stream=[]
        for t in range(14):
            o,state=m.step(tensor[:2,t],state)
            stream.append(o)
        stream=torch.stack(stream,dim=1)
    assert torch.allclose(logits,stream,atol=2e-5,rtol=1e-5)
    save(tmp_path/'weights.pt',m,{'unit_test':True})
    copied,metadata=restore(tmp_path/'weights.pt')
    assert metadata['unit_test']
    assert torch.equal(copied(tensor),m(tensor))
    response=m.generate(sp,'Hiç bilinmeyen bir konu: araştırma',max_new_tokens=25,temperature=0)
    assert isinstance(response['text'],str)
