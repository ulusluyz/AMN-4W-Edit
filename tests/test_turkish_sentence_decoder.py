"""Real-word autoregressive Turkish sentence-generation checks."""
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from relational_qa import ID
from turkish_sentence_decoder import (
    SentenceDecoder, WORD_ID, WORDS, reference_sentence,
    load_frozen
)

def test_turkish_reference_targets_have_full_sentences():
    phrase,ids=reference_sentence(torch.tensor([ID['Yiğit'],ID['renk']]),torch.tensor(ID['sarı']))
    assert phrase=="Yiğit'in kaleminin rengi sarıdır."
    assert len(ids)==5 and int(ids[-1])==WORD_ID['<eos>']
    phrase2,ids2=reference_sentence(torch.tensor([ID['Can'],ID['konum']]),torch.tensor(ID['park']))
    assert phrase2=='Can şu anda parktadır.'
    assert len(ids2)==5

def test_decoder_generates_without_reference_tokens():
    torch.manual_seed(5)
    model=SentenceDecoder()
    query=torch.tensor([[ID['Can'],ID['konum']]])
    logits=torch.randn(1, len(load_frozen('dual_upsert',0).out[-1].bias))
    prediction=model.generate(logits,query)
    assert isinstance(prediction,list) and isinstance(prediction[0],str)

def test_three_frozen_backbones_load_existing_weights():
    for arch in ('amn','dual_upsert','transformer'):
        model=load_frozen(arch,0)
        assert not any(p.requires_grad for p in model.parameters())
