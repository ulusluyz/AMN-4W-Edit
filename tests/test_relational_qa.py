"""CI regression for the structured Turkish relation-learning experiment."""
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from relational_qa import (
    NAMES, RELS, VALUES, FACTS, VOCAB, split_of, generate,
    AssociativeAMN, DualAddressAMN, train
)


def test_partitions_disjoint_and_words_known():
    sets = [set(FACTS[part]) for part in ('train','dev','test')]
    assert list(map(len, sets)) == [264, 25, 31]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    words = set(w for fact in FACTS['train'] for w in fact)
    assert set(NAMES+RELS+VALUES['renk']+VALUES['konum']) <= words
    assert all(split_of(fact) == part for part in FACTS for fact in FACTS[part])


def test_generator_target_is_context_fact_only():
    for part in ('train','dev','test'):
        contexts, queries, labels = generate(part, 64, 4, random.Random(42))
        for facts, query, answer in zip(contexts, queries, labels):
            matches = (facts[:, :2] == query).all(-1)
            assert int(matches.sum()) == 1
            assert int(facts[matches, 2][0]) == int(answer)
            assert len({tuple(x[:2].tolist()) for x in facts}) == 4
            for row in facts:
                assert split_of(tuple(VOCAB[i] for i in row.tolist())) == part


def test_protected_reads_and_deletion():
    torch.manual_seed(8)
    model=DualAddressAMN(48)
    contexts, _, _=generate('train',64,4,random.Random(3))
    keys=model.address(contexts[:,:,:2])
    values=F.normalize(model.value(model.embed(contexts[:,:,2])),dim=-1)
    gram=keys@keys.transpose(-2,-1)
    eye=torch.eye(4)
    dual=torch.linalg.solve(gram+1e-5*eye,keys)
    state=values.transpose(-2,-1)@dual
    recovered=(state@keys.transpose(-2,-1)).transpose(-2,-1)
    assert (recovered-values).abs().max().item()<5e-4
    # Rebuild without the first key; other keys must still read their values.
    k, v=keys[:,1:],values[:,1:]
    d=torch.linalg.solve(k@k.transpose(-2,-1)+1e-5*torch.eye(3),k)
    state2=v.transpose(-2,-1)@d
    r=(state2@k.transpose(-2,-1)).transpose(-2,-1)
    assert (r-v).abs().max().item()<5e-4


def test_gradients_are_finite():
    torch.manual_seed(44)
    contexts,queries,labels=generate('train',16,4,random.Random(4))
    for cls in (AssociativeAMN, DualAddressAMN):
        model=cls(48)
        F.cross_entropy(model(contexts,queries),labels).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())


def test_50_step_dual_heldout_smoke():
    result=train('dual',seed=0,steps=50,batch=128,
                 dev_every=25,dev_size=300,test_size=500,output_dir=None)
    assert result['steps']==50
    assert result['dev']['correct']==300
    assert result['test']['correct']==500
