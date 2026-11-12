"""Document-disjoint UTF-8 corpus preparation; no document-specific model logic."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def document_digest(text):
    normalized = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_documents(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.rglob("*.txt")) + sorted(path.rglob("*.jsonl"))
    else:
        files = [path]
    if not files:
        raise ValueError("No .txt or .jsonl documents found")
    for file in files:
        if file.suffix == ".txt":
            yield str(file), file.read_text(encoding="utf-8")
        elif file.suffix == ".jsonl":
            for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                row = json.loads(line)
                yield f"{file}:{number}", row["text"]
        else:
            raise ValueError("Only .txt and .jsonl input is supported")


def prepare(input_path, output_path, *, val_ratio=0.1, test_ratio=0.1, license_id="UNSPECIFIED"):
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("invalid split ratios")
    records = {}
    for origin, raw in source_documents(input_path):
        text = unicodedata.normalize("NFC", raw).strip()
        if len(text.encode("utf-8")) < 8:
            continue
        digest = document_digest(text)
        records.setdefault(digest, {"sha256": digest, "origin": origin, "text": text})
    if len(records) < 3:
        raise ValueError("At least 3 different documents are required for disjoint splits")
    # Rank by digest, not by byte chunks; exact normalized duplicates are grouped.
    ordered = [records[key] for key in sorted(records)]
    n = len(ordered)
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_test + n_val >= n:
        raise ValueError("Not enough documents to form a training split")
    split_docs = {
        "test": ordered[:n_test],
        "validation": ordered[n_test:n_test + n_val],
        "train": ordered[n_test + n_val:],
    }
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"license": license_id, "split_method": "sorted normalized-document SHA-256",
                "counts": {}, "documents": []}
    for name, docs in split_docs.items():
        with (root / (name + ".jsonl")).open("x", encoding="utf-8") as out:
            for row in docs:
                out.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
                manifest["documents"].append({"sha256": row["sha256"], "origin": row["origin"],
                                              "split": name, "bytes": len(row["text"].encode("utf-8"))})
        manifest["counts"][name] = len(docs)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_split(root, name):
    path = Path(root) / (name + ".jsonl")
    docs = [json.loads(line)["text"].encode("utf-8") for line in path.read_text(encoding="utf-8").splitlines()]
    return [doc for doc in docs if len(doc) >= 2]


def main():
    parser = argparse.ArgumentParser(description="Prepare a document-disjoint raw Turkish text corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--license", default="UNSPECIFIED")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    manifest = prepare(args.input, args.output, val_ratio=args.val_ratio,
                       test_ratio=args.test_ratio, license_id=args.license)
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
