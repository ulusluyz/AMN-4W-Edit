"""Bounded tests for full-corpus completion gate and sequential-learning probe.

These tests verify instrumentation, NOT general intelligence.
"""
import json
import math

import pytest
import torch

from amn_nmc import full_pass
from amn_nmc.continual_probe import measure_seed


def test_full_pass_refuses_incomplete_and_accepts_exact_coverage(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with (corpus / "train.jsonl").open("w", encoding="utf-8") as output:
        for i in range(3):
            output.write(json.dumps({"text": "Türkçe metin " * 5 + str(i)}, ensure_ascii=False) + "\n")
    monkeypatch.setattr(full_pass, "validate_source",
                        lambda data: {"source_sha256": "synthetic_test"})
    from amn_nmc.corpus import load_split
    docs = load_split(corpus, "train")
    expected_bytes = sum(len(doc) - 1 for doc in docs)
    expected_steps = sum(math.ceil((len(doc) - 1) / 16) for doc in docs)
    checkpoint = tmp_path / "coverage.pt"
    base = {
        "format_version": 2,
        "training_settings": {"chunk_size": 16},
        "step": expected_steps - 1,
        "training_bytes": expected_bytes,
        "visited_training_docs": [0, 1, 2],
        "cursor": {"epoch": 0, "doc_position": 2, "byte_offset": 16},
    }
    torch.save(base, checkpoint)
    assert not full_pass.verify(corpus, checkpoint)["one_full_epoch_complete"]
    base["step"] = expected_steps
    base["cursor"] = {"epoch": 0, "doc_position": 3, "byte_offset": 0}
    torch.save(base, checkpoint)
    complete = full_pass.verify(corpus, checkpoint)
    assert complete["one_full_epoch_complete"]
    assert complete["visited_pages"] == 3
    assert complete["expected_first_pass_target_bytes"] == expected_bytes


def test_sequential_learning_measure_reports_forgetting_without_fake_success(tmp_path):
    data = tmp_path / "corpus"
    data.mkdir()
    records = [("Türkçe dil öğrenme ve metin yazma incelemesi. " * 3 + str(i))
               for i in range(26)]
    (data / "train.jsonl").write_text(
        "".join(json.dumps({"text": doc}, ensure_ascii=False) + "\n" for doc in records),
        encoding="utf-8")
    results = measure_seed(data, seed=17, updates=2)
    assert results["updates_per_phase"] == 2
    assert results["phase_a_training_bytes"] > 0
    assert results["phase_b_training_bytes"] > 0
    for key in ("a_forgetting_bpb_delta", "b_learning_bpb_improvement",
                "a_bpb_after_A", "a_bpb_after_B", "b_bpb_before_B",
                "b_bpb_after_B"):
        assert math.isfinite(results[key])
    assert results["a_retained_or_improved"] == (
        results["a_bpb_after_B"] <= results["a_bpb_after_A"])
    assert results["b_adapted"] == (
        results["b_bpb_after_B"] < results["b_bpb_before_B"])
