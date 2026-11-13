"""Deterministic CPU autoregressive training, validation, stateful TBPTT and resume.

No fixed QA labels, document-specific processing, or external fact retrieval.
The underlying byte language objective is independent of its training source.
"""
from __future__ import annotations

import argparse
import hashlib
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
from .state import BlockState, NMCState

STATE_FIELDS = ("hidden", "fast", "keys", "values", "usage", "read")


def chunks(document, length):
    """Give aligned input and next-byte targets, never crossing a document end."""
    if length < 1:
        raise ValueError("chunk length must be positive")
    stream = torch.tensor(list(document), dtype=torch.long)
    for start in range(0, stream.numel() - 1, length):
        stop = min(start + length, stream.numel() - 1)
        yield stream[start:stop].unsqueeze(0), stream[start + 1:stop + 1].unsqueeze(0)


@torch.no_grad()
def evaluate(model, docs, chunk_size, *, max_bytes=None):
    """Evaluate seen tokens exactly once per document. max_bytes is optional CPU budget."""
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    model.eval()
    total_nats, nbytes = 0.0, 0
    for doc in docs:
        state = None
        for x, y in chunks(doc, chunk_size):
            if max_bytes is not None:
                left = max_bytes - nbytes
                if left <= 0:
                    break
                x, y = x[:, :left], y[:, :left]
            logits, state = model(x, state)
            total_nats += float(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum"))
            nbytes += int(y.numel())
        if max_bytes is not None and nbytes >= max_bytes:
            break
    if nbytes == 0:
        raise ValueError("No evaluation bytes available")
    return {"bpb": total_nats / (nbytes * math.log(2)),
            "nats_per_byte": total_nats / nbytes, "bytes": nbytes}


def atomic_save(payload, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".nmc_", suffix=".tmp", dir=filename.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, filename)
    finally:
        temp.unlink(missing_ok=True)


def save_state(state):
    if state is None:
        return None
    return [{key: getattr(block, key).detach().clone().cpu() for key in STATE_FIELDS}
            for block in state.blocks]


def load_state(saved):
    if saved is None:
        return None
    return NMCState(tuple(BlockState(*(item[key] for key in STATE_FIELDS)) for item in saved))


def _fingerprints(root):
    return {split: hashlib.sha256((Path(root) / (split + ".jsonl")).read_bytes()).hexdigest()
            for split in ("train", "validation", "test")}


def _make_order(seed, epoch, length):
    indices = list(range(length))
    random.Random(seed + 7919 * epoch).shuffle(indices)
    return indices


def _atomic_json(data, filename):
    target = Path(filename)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".log_", suffix=".tmp",
                                     dir=target.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, target)


def train(data, output, config_path, seed=17, *, resume=False, extra_steps=0):
    """Train NMC, saving best and resumable last checkpoint.

    "resume" continues the exact next (document, byte offset) in the fixed order,
    carrying the detached per-document memory state and Adam moments.
    "extra_steps" extends the saved step budget without restarting training.
    """
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    mc = NMCConfig(**configuration["model"])
    settings = configuration["training"]
    if settings.get("device", "cpu") != "cpu":
        raise ValueError("CPU-only training runner")
    if extra_steps < 0:
        raise ValueError("extra_steps must be nonnegative")
    torch.set_num_threads(int(settings.get("threads", 2)))
    random.seed(seed)
    torch.manual_seed(seed)
    train_docs = load_split(data, "train")
    val_docs = load_split(data, "validation")
    if not train_docs or not val_docs:
        raise ValueError("train and validation require documents")
    chunk_size = int(settings.get("chunk_size", 64))
    epochs = int(settings.get("epochs", 1))
    maximum = int(settings.get("max_steps", 100)) + extra_steps
    eval_every = int(settings.get("eval_every", 0))
    budget = settings.get("eval_max_bytes", None)
    if budget is not None:
        budget = int(budget)
    if chunk_size < 1 or epochs < 1 or maximum < 1 or eval_every < 0:
        raise ValueError("invalid training schedule")
    out = Path(output)
    if resume:
        if not (out / "last.pt").is_file():
            raise FileNotFoundError("Resume requires last.pt")
    else:
        out.mkdir(parents=True, exist_ok=False)
    model = AMNNMC(mc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get("lr", 0.0003),
                                  betas=(0.9, 0.95), weight_decay=settings.get("weight_decay", 0.01))
    fingerprints = _fingerprints(data)
    parameters = sum(p.numel() for p in model.parameters())
    progress = {"epoch": 0, "doc_position": 0, "byte_offset": 0}
    state = None
    step, training_bytes, history = 0, 0, []
    visited_docs = set()
    initial_validation = None
    best_score = float("inf")
    if resume:
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if saved.get("format_version") != 2 or saved.get("seed") != seed:
            raise ValueError("Incompatible checkpoint format or seed")
        saved_config = dict(saved.get("config", {}))
        saved_config.setdefault("addressing", "legacy")
        if saved_config != asdict(mc) or saved.get("fingerprints") != fingerprints:
            raise ValueError("Changed corpus or architecture on resume")
        previous_settings = saved["training_settings"]
        # Extending max_steps is allowed but changing causal TBPTT or optimizer is not.
        for key in ("chunk_size", "lr", "threads", "weight_decay", "epochs", "eval_every", "eval_max_bytes"):
            if previous_settings.get(key) != settings.get(key):
                raise ValueError("Changing training settings on resume: " + key)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        progress = saved["cursor"]
        state = load_state(saved.get("recurrent_state"))
        step = int(saved["step"])
        training_bytes = int(saved["training_bytes"])
        visited_docs = set(saved.get("visited_training_docs", []))
        history = list(saved["history"])
        initial_validation = saved["initial_validation"]
        best_score = float(saved["best_validation"]["bpb"])
        if maximum <= step:
            raise ValueError("Requested max_steps is not greater than resumed step count")
    else:
        initial_validation = evaluate(model, val_docs, chunk_size, max_bytes=budget)
        best_score = initial_validation["bpb"]
        (out / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")

    current_validation = initial_validation
    best_validation = {"bpb": best_score}

    def build_payload():
        return {
            "format_version": 2, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "config": asdict(mc),
            "training_settings": settings, "fingerprints": fingerprints,
            "seed": seed, "step": step, "training_bytes": training_bytes,
            "cursor": dict(progress), "recurrent_state": save_state(state),
            "visited_training_docs": sorted(visited_docs),
            "history": list(history), "initial_validation": initial_validation,
            "best_validation": dict(best_validation), "current_validation": current_validation,
        }

    if not resume:
        atomic_save(build_payload(), out / "initial.pt")
        atomic_save(build_payload(), out / "best.pt")

    last_step = step
    for epoch in range(progress["epoch"], epochs):
        order = _make_order(seed, epoch, len(train_docs))
        first_position = progress["doc_position"] if epoch == progress["epoch"] else 0
        for pos in range(first_position, len(order)):
            doc = train_docs[order[pos]]
            offset = progress["byte_offset"] if epoch == progress["epoch"] and pos == first_position else 0
            if offset == 0:
                state = None
            if offset >= len(doc) - 1:
                continue
            bytes_tensor = torch.tensor(list(doc), dtype=torch.long)
            for start in range(offset, len(doc) - 1, chunk_size):
                stop = min(start + chunk_size, len(doc) - 1)
                x = bytes_tensor[start:stop].unsqueeze(0)
                y = bytes_tensor[start + 1:stop + 1].unsqueeze(0)
                model.train()
                logits, state = model(x, state)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"nonfinite gradient at step {step}")
                optimizer.step()
                state = state.detach()  # TBPTT boundary, not every token.
                step += 1
                visited_docs.add(order[pos])
                training_bytes += y.numel()
                next_offset = stop
                if next_offset >= len(doc) - 1:
                    progress = {"epoch": epoch, "doc_position": pos + 1, "byte_offset": 0}
                    state = None
                else:
                    progress = {"epoch": epoch, "doc_position": pos, "byte_offset": next_offset}
                history.append({"step": step, "train_bpb": float(loss.detach() / math.log(2)),
                                "training_bytes": training_bytes, "gradient_norm": float(grad_norm.detach())})
                if eval_every > 0 and step % eval_every == 0:
                    current_validation = evaluate(model, val_docs, chunk_size, max_bytes=budget)
                    history[-1]["validation_bpb"] = current_validation["bpb"]
                    if current_validation["bpb"] < best_score:
                        best_score = current_validation["bpb"]
                        best_validation = dict(current_validation)
                        atomic_save(build_payload(), out / "best.pt")
                    atomic_save(build_payload(), out / "last.pt")
                if step >= maximum:
                    break
            if step >= maximum:
                break
        if step >= maximum:
            break
        progress = {"epoch": epoch + 1, "doc_position": 0, "byte_offset": 0}
        state = None

    if step == last_step:
        raise RuntimeError("Training ran zero steps; adjust epochs or max_steps")
    current_validation = evaluate(model, val_docs, chunk_size, max_bytes=budget)
    if current_validation["bpb"] < best_score:
        best_score = current_validation["bpb"]
        best_validation = dict(current_validation)
        atomic_save(build_payload(), out / "best.pt")
    atomic_save(build_payload(), out / "last.pt")
    full_validation = evaluate(model, val_docs, chunk_size)
    stats = {
        "seed": seed, "parameter_count": parameters, "steps": step,
        "training_bytes": training_bytes, "initial_validation": initial_validation,
        "final_validation": full_validation, "best_validation_budgeted": best_validation,
        "eval_max_bytes": budget, "best_checkpoint": "best.pt",
        "checkpoint_format": 2,
        "available_training_pages": len(train_docs), "visited_training_pages": len(visited_docs),
        "distinct_validation_pages": len(val_docs),
        "training_corpus_sha256": fingerprints["train"],
        "validation_corpus_sha256": fingerprints["validation"],
        "resumed": resume,
    }
    _atomic_json(stats, out / "summary.json")
    with (out / "training.jsonl").open("w", encoding="utf-8") as logfile:
        for row in history:
            logfile.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train general-purpose byte-level AMN-NMC on CPU")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/nmc/cpu_initial.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra-steps", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output, args.config, args.seed,
                           resume=args.resume, extra_steps=args.extra_steps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
