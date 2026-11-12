"""Verify new native AMN epoch sampler prevents silent undercoverage."""
import sys
from pathlib import Path

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from train_amn_tm_v3 import BalancedEpochSampler


def test_every_question_is_visited_before_repetition():
    rows=[{"prompt":str(i),"answer":"Doğru cevap "+str(i)} for i in range(37)]
    sampler=BalancedEpochSampler(rows,seed=100)
    outputs=[]
    for _ in range(10):
        batch=sampler.take(4)
        assert len(set(x["prompt"] for x in batch))==4
        outputs.extend(x["prompt"] for x in batch)
    assert set(outputs)=={str(i) for i in range(37)}
    stats=sampler.stats()
    assert stats["dataset_size"]==37
    assert stats["unique_samples"]==37
    assert stats["minimum_exposures"]>=1
    assert stats["maximum_exposures"]<=2
    assert stats["total_samples"]==40


def test_epoch_sampling_is_deterministic_for_same_seed():
    rows=[{"prompt":str(i)} for i in range(19)]
    a=BalancedEpochSampler(rows,seed=55)
    b=BalancedEpochSampler(rows,seed=55)
    for _ in range(20):
        assert [r["prompt"] for r in a.take(5)]==[r["prompt"] for r in b.take(5)]
    assert a.stats()==b.stats()


def test_coverage_counts_all_never_sampled_questions():
    rows=[{"prompt":str(i)} for i in range(100)]
    sampler=BalancedEpochSampler(rows,seed=2)
    sampler.take(4)
    stat=sampler.stats()
    assert stat["total_samples"]==4
    assert stat["unique_samples"]==4
    assert stat["minimum_exposures"]==0
    assert stat["coverage_ratio"]==0.04
    with pytest.raises(ValueError):
        sampler.take(101)
