"""CPU regression tests for UTF-8-safe generation and persisted neural state."""
from pathlib import Path

import pytest
import torch

from amn_nmc import AMNNMC, NMCConfig
from amn_nmc.generate import generate
from amn_nmc.session_memory import load_session, model_sha256, save_session
from amn_nmc.capability_probes import challenge_set, probe


@pytest.fixture
def model():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    return AMNNMC(NMCConfig(d_model=16, memory_dim=8, slots=3, layers=1)).eval()


def test_utf8_strict_generation_all_lengths_and_sampling_modes(model):
    for length in (0, 1, 2, 3, 4, 5, 11, 32):
        for top_k in (0, 1, 5, 40):
            value = generate(model, "Türkçe", new_bytes=length,
                             temperature=0.7, top_k=top_k, seed=17)
            assert isinstance(value, str)
            assert value.startswith("Türkçe")
            assert len(value.encode("utf-8")) == len("Türkçe".encode("utf-8")) + length
            assert "\ufffd" not in value


def test_utf8_valid_across_seeds(model):
    for seed in (0, 1, 17, 29, 43):
        a = generate(model, "Ankara'da ", new_bytes=17, seed=seed)
        b = generate(model, "Ankara'da ", new_bytes=17, seed=seed)
        assert a == b
        assert a.encode("utf-8").decode("utf-8") == a


def test_recurrent_session_roundtrip_and_weight_binding(tmp_path, model):
    input_ = torch.tensor([[65, 66, 195, 167]], dtype=torch.long)
    _, memory = model(input_)
    saved = tmp_path / "session.pt"
    digest = save_session(saved, model, memory.detach(), tokens_seen=4)
    restored, count = load_session(saved, model)
    assert count == 4
    assert digest == model_sha256(model)
    with torch.no_grad():
        a, _ = model(torch.tensor([[68, 69]], dtype=torch.long), memory.detach())
        b, _ = model(torch.tensor([[68, 69]], dtype=torch.long), restored)
    assert torch.equal(a, b)

    with torch.no_grad():
        model.head.bias[0].add_(1)
    with pytest.raises(ValueError, match="does not match"):
        load_session(saved, model)


def test_capability_probes_are_counterfactually_paired(model):
    cases = challenge_set()
    assert len(cases) == 12
    for i in range(6):
        assert cases[i]["correct"] == cases[i + 6]["wrong"]
        assert cases[i]["wrong"] == cases[i + 6]["correct"]
    result = probe(model, max_cases=2)
    assert result["total"] == 2
    assert result["chance_binary"] == 0.5
    assert result["zero_training_from_probes"] is True
