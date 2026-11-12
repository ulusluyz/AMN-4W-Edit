"""Configuration for the CPU-first AMN Neural Memory Core."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NMCConfig:
    vocab_size: int = 256
    d_model: int = 96
    memory_dim: int = 48
    slots: int = 32
    layers: int = 4
    expansion: int = 2
    use_fast: bool = True
    use_episodic: bool = True

    def __post_init__(self):
        if self.vocab_size != 256:
            raise ValueError("The first NMC implementation models raw bytes (256 symbols)")
        for key in ("d_model", "memory_dim", "slots", "layers", "expansion"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
