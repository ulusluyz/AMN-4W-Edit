"""Free-form autoregressive byte generation with strictly valid UTF-8 decoding.

UTF-8 masks constrain byte *encoding*, not learned Turkish semantics, grammar or
facts. The neural model still predicts each byte without hard-coded responses.
"""
import argparse
from pathlib import Path
import torch
from .config import NMCConfig
from .model import AMNNMC


_ASCII = tuple([9, 10] + list(range(32, 127)))


def _lead_choices(remaining):
    """Legal first bytes of a Unicode scalar value fitting remaining bytes."""
    choices = list(_ASCII)
    if remaining >= 2:
        choices.extend(range(0xC2, 0xE0))
    if remaining >= 3:
        choices.extend(range(0xE0, 0xF0))
    if remaining >= 4:
        choices.extend(range(0xF0, 0xF5))
    return choices


def _following_ranges(lead):
    if 0xC2 <= lead <= 0xDF:
        return [(0x80, 0xBF)]
    if lead == 0xE0:
        return [(0xA0, 0xBF), (0x80, 0xBF)]
    if 0xE1 <= lead <= 0xEC or 0xEE <= lead <= 0xEF:
        return [(0x80, 0xBF), (0x80, 0xBF)]
    if lead == 0xED:
        return [(0x80, 0x9F), (0x80, 0xBF)]  # exclude UTF-16 surrogates
    if lead == 0xF0:
        return [(0x90, 0xBF), (0x80, 0xBF), (0x80, 0xBF)]
    if 0xF1 <= lead <= 0xF3:
        return [(0x80, 0xBF)] * 3
    if lead == 0xF4:
        return [(0x80, 0x8F), (0x80, 0xBF), (0x80, 0xBF)]  # max U+10FFFF
    return []


@torch.inference_mode()
def generate(model, prompt, *, new_bytes=128, temperature=0.8, top_k=40,
             seed=17, chunk_size=64):
    if not prompt:
        raise ValueError('Nonempty UTF-8 prompt is required')
    if not isinstance(new_bytes, int) or new_bytes < 0 or temperature <= 0 or top_k < 0:
        raise ValueError('Invalid generation arguments')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    data = prompt.encode('utf-8', errors='strict')
    model.eval()
    state = None
    logits = None
    for i in range(0, len(data), chunk_size):
        inp = torch.tensor(list(data[i:i + chunk_size]), dtype=torch.long).unsqueeze(0)
        logits, state = model(inp, state)
    output = bytearray(data)
    pending = []
    for t in range(new_bytes):
        left = new_bytes - t
        allowed = (list(range(pending[0][0], pending[0][1] + 1))
                   if pending else _lead_choices(left))
        index_tensor = torch.tensor(allowed, dtype=torch.long)
        scores = logits[0, -1].float().index_select(0, index_tensor) / temperature
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite byte logits')
        if top_k:
            best_scores, selected = torch.topk(scores, min(top_k, scores.numel()))
            options = index_tensor[selected]
            probs = torch.softmax(best_scores, dim=-1)
        else:
            options = index_tensor
            probs = torch.softmax(scores, dim=-1)
        new_byte = int(options[torch.multinomial(probs, 1, generator=generator)])
        output.append(new_byte)
        if pending:
            low, high = pending.pop(0)
            assert low <= new_byte <= high
        elif new_byte >= 0x80:
            pending = _following_ranges(new_byte)
        tokens = torch.tensor([[new_byte]], dtype=torch.long)
        logits, state = model(tokens, state)
    if pending:
        raise AssertionError('UTF-8 output ended with an incomplete sequence')
    return bytes(output).decode('utf-8', errors='strict')


def main():
    parser = argparse.ArgumentParser(description='UTF-8-safe autoregressive AMN-NMC sampling')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--new-bytes', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    checkpoint = torch.load(Path(args.checkpoint), map_location='cpu', weights_only=True)
    model = AMNNMC(NMCConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'])
    print(generate(model, args.prompt, new_bytes=args.new_bytes,
                   temperature=args.temperature, top_k=args.top_k, seed=args.seed))


if __name__ == '__main__':
    main()
