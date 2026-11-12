"""Unconstrained autoregressive byte generation from an AMN-NMC checkpoint."""
import argparse
from pathlib import Path
import torch
from .config import NMCConfig
from .model import AMNNMC


@torch.no_grad()
def generate(model, prompt, *, new_bytes=128, temperature=0.8, top_k=40, seed=17, chunk_size=64):
    if not prompt:
        raise ValueError("A nonempty UTF-8 prompt is required")
    if temperature <= 0 or new_bytes < 0 or top_k < 0:
        raise ValueError("invalid generation arguments")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    data = prompt.encode("utf-8")
    model.eval()
    state = None
    logits = None
    for i in range(0, len(data), chunk_size):
        tokens = torch.tensor(list(data[i:i + chunk_size]), dtype=torch.long).unsqueeze(0)
        logits, state = model(tokens, state)
    output = bytearray(data)
    for _ in range(new_bytes):
        scores = logits[0, -1] / temperature
        if top_k:
            values, indices = torch.topk(scores, min(top_k, scores.numel()))
            probs = torch.softmax(values, -1)
            index = int(indices[torch.multinomial(probs, 1, generator=generator)])
        else:
            index = int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator))
        output.append(index)
        logits, state = model(torch.tensor([[index]], dtype=torch.long), state)
    return bytes(output).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Sample raw UTF-8 bytes from trained AMN-NMC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--new-bytes", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    model = AMNNMC(NMCConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    print(generate(model, args.prompt, new_bytes=args.new_bytes,
                   temperature=args.temperature, top_k=args.top_k, seed=args.seed))


if __name__ == "__main__":
    main()
