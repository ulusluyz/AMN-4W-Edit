"""CPU-first, state-carry truncated BPTT training on general UTF-8 text."""
import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from .config import NMCConfig
from .corpus import load_split
from .model import AMNNMC


def chunks(document, length):
    """Each observed byte predicts the next byte; carries state across chunks."""
    stream = torch.tensor(list(document), dtype=torch.long)
    for start in range(0, stream.numel() - 1, length):
        yield stream[start:min(start + length, stream.numel() - 1)].unsqueeze(0), stream[start + 1:min(start + length + 1, stream.numel())].unsqueeze(0)


@torch.no_grad()
def evaluate(model, docs, chunk_size):
    model.eval()
    total, nbytes = 0.0, 0
    for doc in docs:
        state = None
        for x, y in chunks(doc, chunk_size):
            logits, state = model(x, state)
            total += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum").item()
            nbytes += y.numel()
    if not nbytes:
        raise ValueError("No evaluation bytes available")
    return {"bpb": total / (nbytes * math.log(2)), "nats_per_byte": total / nbytes,
            "bytes": nbytes}


def atomic_save(payload, filename):
    filename = Path(filename)
    with tempfile.NamedTemporaryFile(prefix=".nmc_", suffix=".tmp", dir=filename.parent, delete=False) as file:
        temp = Path(file.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, filename)
    finally:
        if temp.exists():
            temp.unlink()


def train(data, output, config_path, seed=17):
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_config = NMCConfig(**configuration["model"])
    settings = configuration["training"]
    if settings.get("device", "cpu") != "cpu":
        raise ValueError("The first verified training runner intentionally uses CPU")
    torch.set_num_threads(settings.get("threads", 2))
    random.seed(seed)
    torch.manual_seed(seed)
    train_docs = load_split(data, "train")
    val_docs = load_split(data, "validation")
    if not train_docs or not val_docs:
        raise ValueError("Both train and validation splits need nonempty documents")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    model = AMNNMC(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get("lr", 0.0003),
                                  betas=(0.9, 0.95), weight_decay=0.01)
    chunk_size = settings.get("chunk_size", 64)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    max_steps = settings.get("max_steps", 100)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    stats = {"seed": seed, "parameter_count": sum(p.numel() for p in model.parameters()),
             "initial_validation": evaluate(model, val_docs, chunk_size), "steps": 0}
    best = stats["initial_validation"]["bpb"]
    def save(name, metrics):
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": asdict(model_config), "seed": seed, "step": stats["steps"],
                     "metrics": metrics}, output / name)
    save("initial.pt", stats["initial_validation"])
    rng = random.Random(seed)
    log = []
    for epoch in range(settings.get("epochs", 1)):
        docs = list(train_docs)
        rng.shuffle(docs)
        for doc in docs:
            state = None  # Reset at document boundary, never leak between documents.
            for x, y in chunks(doc, chunk_size):
                model.train()
                logits, state = model(x, state)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                state = state.detach()  # TBPTT truncation ONLY at chunk boundary.
                stats["steps"] += 1
                log.append({"step": stats["steps"], "train_bpb": float(loss.item() / math.log(2)),
                            "gradient_norm": float(norm)})
                if stats["steps"] >= max_steps:
                    break
            if stats["steps"] >= max_steps:
                break
        if stats["steps"] >= max_steps:
            break
    stats["final_validation"] = evaluate(model, val_docs, chunk_size)
    stats["best_validation"] = min(best, stats["final_validation"]["bpb"])
    if stats["final_validation"]["bpb"] < best:
        save("best.pt", stats["final_validation"])
        stats["best_checkpoint"] = "best.pt"
    else:
        stats["best_checkpoint"] = "initial.pt"
    save("last.pt", stats["final_validation"])
    (output / "training.jsonl").write_text(''.join(json.dumps(row) + "\n" for row in log))
    (output / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train AMN-NMC from scratch, CPU only")
    parser.add_argument("--data", required=True, help="directory containing train.jsonl, validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/nmc/cpu_initial.json")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output, args.config, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
