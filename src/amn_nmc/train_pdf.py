"""Run CPU AMN-NMC training only from the one user-supplied PDF fingerprint.

This protects against accidentally selecting the previous Wikipedia training data.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .pdf_corpus import SOURCE_SHA256
from .train import train


def validate_source(data):
    root = Path(data)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("dataset_kind") != "single_academic_pdf_internal_research"
            or manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("source_document_count") != 1
            or manifest.get("split_type") != "same_document_page_holdout; NOT independent validation"):
        raise ValueError("Training source is not the explicitly approved PDF")
    hashes = manifest.get("split_files_sha256", {})
    for split in ("train", "validation", "test"):
        expected = hashes.get(split)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Missing split hash: " + split)
        got = hashlib.sha256((root / (split + ".jsonl")).read_bytes()).hexdigest()
        if got != expected:
            raise ValueError("Modified training partition: " + split)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="PDF-source-locked AMN-NMC CPU run")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/nmc/cpu_smoke.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra-steps", type=int, default=0)
    args = parser.parse_args()
    manifest = validate_source(args.data)
    result = train(args.data, args.output, args.config, args.seed,
                   resume=args.resume, extra_steps=args.extra_steps)
    result["training_source_sha256"] = manifest["source_sha256"]
    result["evaluation_limitation"] = manifest["split_type"]
    Path(args.output, "pdf_run_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
