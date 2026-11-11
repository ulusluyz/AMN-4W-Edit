"""Acceptance tests for parameter-compatible Dual AMN last-write-wins.

Run: python -m pytest -q tests/test_dual_upsert.py
"""
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from dual_upsert import DualAddressUpsertAMN, last_write_mask
from relational_qa import DualAddressAMN, generate, ID, VOCAB
from validate_three_models_turkish import make_last_write_batch


def test_last_write_mask_exact_and_batch_specific():
    # Row0 has three writes to key (Ali, renk); row1 has no collisions.
    pairs=[
        [[ID["Ali"],ID["renk"],ID["mavi"]],
         [ID["Can"],ID["konum"],ID["ev"]],
         [ID["Ali"],ID["renk"],ID["sarı"]],
         [ID["Ali"],ID["renk"],ID["yeşil"]]],
        [[ID["Ali"],ID["renk"],ID["mavi"]],
         [ID["Can"],ID["konum"],ID["ev"]],
         [ID["Ece"],ID["renk"],ID["beyaz"]],
         [ID["Ali"],ID["konum"],ID["park"]]],
    ]
    out=last_write_mask(torch.tensor(pairs))
    assert out.tolist()==[[False,True,False,True],[True,True,True,True]]


def test_original_logits_and_gradients_identical_with_unique_pairs():
    torch.manual_seed(12)
    old=DualAddressAMN(48)
    new=DualAddressUpsertAMN(48)
    new.load_state_dict(old.state_dict())
    facts,query,answers=generate("train",16,6,random.Random(10))
    a=old(facts,query);b=new(facts,query)
    assert torch.equal(a,b)
    F.cross_entropy(a,answers).backward()
    F.cross_entropy(b,answers).backward()
    for (n,p),(m,q) in zip(old.named_parameters(),new.named_parameters()):
        assert n==m
        assert torch.equal(p.grad,q.grad),n


def test_repeated_key_rank_stability_and_finite_gradients():
    torch.manual_seed(99)
    model=DualAddressUpsertAMN(48)
    x,query,answers,_=make_last_write_batch(0,32)
    assert bool(torch.isfinite(model(x,query)).all())
    loss=F.cross_entropy(model(x,query),answers)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_batched_exact_last_write_with_pretrained_weights_smoke():
    ckpt=ROOT/"results/relational_qa/equal50/dual_0.pt"
    saved=torch.load(ckpt,weights_only=True,map_location="cpu")
    old=DualAddressAMN(48); new=DualAddressUpsertAMN(48)
    old.load_state_dict(saved["state_dict"],strict=True)
    new.load_state_dict(saved["state_dict"],strict=True)
    old.eval(); new.eval()
    x,q,y,_=make_last_write_batch(0,128)
    with torch.no_grad():
        new_correct=int((new(x,q).argmax(-1)==y).sum())
        old_correct=int((old(x,q).argmax(-1)==y).sum())
    assert new_correct==128
    assert new_correct>old_correct
