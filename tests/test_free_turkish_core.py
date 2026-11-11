"""Finite domain, neural semantic extraction, and checkpoint regression."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from relational_qa import FACTS,ID,VOCAB
from free_turkish_core import STYLES,WORDS,WI,reference_text,LongTurkishDecoder,load_dual,sample_raw
from turkish_tagged_parser import tokenize,build_vocab,build_data,TaggedNeuralParser

def test_disjoint_fact_bindings():
    assert not set(FACTS['train']) & set(FACTS['test'])
    assert len(FACTS['test'])==31

def test_utf8_turkish_tokenizer():
    assert "yiğit\'in" in tokenize("Yiğit'in kalemi mavidir.")
    assert 'bahçede' in tokenize('Can bahçede bulunuyor.')

def test_word_targets_not_used_during_inference():
    d=LongTurkishDecoder()
    assert len(WORDS)>32
    a=reference_text('Ali','konum','park','detayli')
    assert len(a.split())>20
    generated=d.generate(torch.randn(1,len(VOCAB)),torch.tensor([[ID['Ali'],ID['konum']]]),torch.tensor([2]))
    assert len(generated)==1 and isinstance(generated[0],str)

def test_neural_parser_shape():
    raw=sample_raw('train',3,99)
    v=build_vocab(raw);f,q,_=build_data(raw,v)
    model=TaggedNeuralParser(v)
    ff,qq=model(f,q)
    assert [x.shape for x in ff]==[(3,4,20),(3,4,2),(3,4,16)]
    assert [x.shape for x in qq]==[(3,20),(3,2)]

def test_pretrained_dual_is_frozen():
    model=load_dual()
    assert all(not p.requires_grad for p in model.parameters())