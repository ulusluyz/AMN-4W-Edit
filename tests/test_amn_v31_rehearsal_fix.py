"""Regression checks for controlled AMN v3.1 seen-question repair."""
import sys
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))

from amn_tm_v31 import AMNTMv31
from amn_tm_v3 import build_sft,collate_lm,IGNORE
from fix_amn_v31_rehearsal import char_f1,normalize,teacher_forced,summarize_outputs

class DummyTokenizer:
    def bos_id(self):return 1
    def eos_id(self):return 2
    def pad_id(self):return 0
    def encode(self,text):
        return [4+b%100 for b in text.encode("utf-8")]

def test_prefix_labels_masked_and_first_answer_is_supervised():
    sp=DummyTokenizer()
    q="Dünyada kaç kuş türü var?"
    a="Yaklaşık 10 bin tür vardır."
    prefix=[sp.bos_id()]+sp.encode("Kullanıcı: "+q+"\nAsistan: ")
    x,y=build_sft(sp,q,a,192)
    assert y[:len(prefix)-1]==[IGNORE]*(len(prefix)-1)
    assert y[len(prefix)-1]==sp.encode(a)[0]
    assert y[-1]==sp.eos_id()
    assert len(x)==len(y)
    assert not (set(x)&{IGNORE})

def test_reference_nll_and_gradients_on_real_autoregressive_loss():
    sp=DummyTokenizer()
    torch.manual_seed(5)
    model=AMNTMv31(vocab_size=128,width=32,layers=1,key_dim=8,semantic_dim=16,pad_id=0)
    row=build_sft(sp,"Bir soru nedir?","Gerçek bir cevaptır.",192)
    x,y=collate_lm([row],0)
    initial=teacher_forced(model,[({"question":"Bir soru nedir?","training_reference":"Gerçek bir cevaptır."},row)],0)[0]
    assert initial>0 and initial<20
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.003)
    for step in range(15):
        logits=model(x)
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,128),y.reshape(-1),ignore_index=IGNORE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after=teacher_forced(model,[({"question":"Bir soru nedir?","training_reference":"Gerçek bir cevaptır."},row)],0)[0]
    assert after<initial

def test_generation_summary_measures_exact_not_semantic_accuracy():
    assert normalize("  Merhaba    Dünya ")==normalize("merhaba dünya")
    assert char_f1("doğru","doğru")==1
    a=summarize_outputs([{"exact":True,"char_f1_proxy":1},
                         {"exact":False,"char_f1_proxy":0.2}])
    assert a["exact_count"]==1
    assert a["avg_char_f1_proxy"]==0.6
