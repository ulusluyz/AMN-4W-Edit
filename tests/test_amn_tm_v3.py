"""Test AMN-TM v3 model math, causal memory and all training objectives."""
import random
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))

from amn_tm_v3 import (
    AMNTMv3,INTENTS,build_sft,build_answer_lm,collate_lm,
    collate_questions,parameters,save,restore,IGNORE
)
from train_amn_tm_v3 import intent_label,multi_turn_pair

class DummySP:
    def bos_id(self):return 1
    def eos_id(self):return 2
    def pad_id(self):return 0
    def encode(self,s,out_type=None):
        return [3+(b % 90) for b in s.encode("utf-8")]

def test_target_16m_configuration():
    model=AMNTMv3(8192,544,6,128,192,0)
    assert 15_000_000 <= parameters(model) <= 18_000_000
    assert not any(isinstance(x,torch.nn.MultiheadAttention) for x in model.modules())
    assert model.config["vocab_size"]==8192

def test_turkish_utf8_no_answer_leakage_and_multi_turn():
    sp=DummySP()
    sample=build_sft(sp,"Öğrenmek nedir?","Bilgi edinmektir.",max_len=160)
    assert sample is not None
    x,y=sample
    answer=list("Bilgi edinmektir.".encode("utf-8"))
    assert y.count(IGNORE)>8
    assert y[-1]==sp.eos_id()
    assert x[-1] != sp.eos_id()
    assert build_answer_lm(sp,"Türkçe bir cevap",160) is not None
    a={"prompt":"Nasılsın?","answer":"İyiyim."}
    b={"prompt":"Merhaba?","answer":"Selam!"}
    assert multi_turn_pair(sp,a,b,256) is not None

def test_all_heads_train_and_state_round_trip(tmp_path):
    torch.manual_seed(8)
    sp=DummySP()
    model=AMNTMv3(vocab_size=128,width=32,layers=1,key_dim=8,semantic_dim=16,pad_id=0)
    rows=[build_sft(sp,"Bilgi nedir?","Bilgi öğrenmektir.",144),
          build_sft(sp,"Nasıl yapılır?","Önce hazırlık gerekir.",144)]
    ids,gold=collate_lm(rows,0)
    logits=model(ids)
    assert logits.shape==(2,ids.shape[1],128)
    lm=F.cross_entropy(logits.reshape(-1,128),gold.reshape(-1),ignore_index=IGNORE)
    q=collate_questions(sp,["Bilgi nedir?","Nasıl yapılır?"],max_len=64)
    evidence=collate_questions(sp,["Bilgi öğrenmektir.","Önce hazırlık gerekir."],max_len=64)
    intentions=torch.tensor([intent_label("Bilgi nedir?"),intent_label("Nasıl yapılır?")])
    loss=lm+0.2*F.cross_entropy(model.intent_logits(q),intentions)
    loss=loss+0.2*model.retrieval_loss(q,evidence)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    out=tmp_path/"test.pt"
    save(out,model,{"checkpoint":"pilot"})
    clone,meta=restore(out)
    assert meta["checkpoint"]=="pilot"
    clone.eval();model.eval()
    with torch.no_grad():
        assert torch.equal(model(ids),clone(ids))

def test_forward_and_step_causal_parity():
    torch.manual_seed(3)
    model=AMNTMv3(128,32,1,8,16,0)
    tokens=torch.randint(4,128,(2,11))
    model.eval()
    with torch.no_grad():
        logits=model(tokens)
        state=None;results=[]
        for i in range(tokens.shape[1]):
            one,state=model.step(tokens[:,i],state)
            results.append(one)
    streamed=torch.stack(results,1)
    assert torch.allclose(logits,streamed,atol=2e-5,rtol=1e-5)

def test_heuristic_labels_are_explicit_not_gold():
    assert INTENTS[intent_label("Lütfen sonucu hesapla")]=="calculation"
    assert INTENTS[intent_label("İki durumu karşılaştır")]=="comparison"
    assert INTENTS[intent_label("Önce nasıl yapılır?")]=="procedure"
    assert INTENTS[intent_label("Nedir?")]=="explanation"
