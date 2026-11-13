"""Check backward-compatible causal NMC addressing, without external training files."""
import torch
from amn_nmc.config import NMCConfig
from amn_nmc.model import AMNNMC

def test_legacy_and_competitive_share_weight_shapes():
    torch.manual_seed(17)
    original = AMNNMC(NMCConfig(d_model=16, memory_dim=8, slots=4, layers=1))
    other = AMNNMC(NMCConfig(d_model=16, memory_dim=8, slots=4, layers=1,
                            addressing="competitive"))
    assert other.load_state_dict(original.state_dict()).missing_keys == []
    assert original.config.addressing == "legacy"
    assert set(other.state_dict()) == set(original.state_dict())

def test_competitive_mode_is_causal_and_bptt_finite():
    torch.manual_seed(29)
    model = AMNNMC(NMCConfig(d_model=16, memory_dim=8, slots=4,
                            layers=1, addressing="competitive"))
    x = torch.randint(0, 256, (1, 21))
    full, _ = model(x)
    a, state = model(x[:, :8])
    b, state = model(x[:, 8:], state)
    assert torch.allclose(full, torch.cat((a, b), dim=1), atol=1e-5)
    assert torch.isfinite(state.blocks[0].fast).all()
    assert torch.isfinite(state.blocks[0].values).all()

def test_competitive_memory_receives_loss_gradients():
    torch.manual_seed(31)
    model = AMNNMC(NMCConfig(d_model=16, memory_dim=8, slots=4,
                            layers=1, addressing="competitive"))
    x = torch.randint(0, 256, (1, 22))
    y, _ = model(x[:, :-1])
    loss = torch.nn.functional.cross_entropy(y.reshape(-1, 256), x[:, 1:].reshape(-1))
    loss.backward()
    assert model.blocks[0].long_write.weight.grad.abs().sum() > 0
    assert model.blocks[0].fast_write.weight.grad.abs().sum() > 0

def test_old_config_is_implicitly_legacy():
    old = {"vocab_size": 256, "d_model": 48, "memory_dim": 24,
           "slots": 8, "layers": 2, "expansion": 2,
           "use_fast": True, "use_episodic": True}
    assert NMCConfig(**old).addressing == "legacy"
