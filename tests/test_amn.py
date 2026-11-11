import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.amn4w import AMN4W

def test_amn4w_forward():
    model = AMN4W(vocab_size=10, embed_dim=16, memory_size=16, max_seq_len=20)
    x = torch.randint(0, 10, (2, 5))
    logits, mem = model(x)
    assert logits.shape == (2, 5, 10)
    assert mem.shape == (2, 16, 16)
