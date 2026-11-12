"""Prepare *only* the explicitly approved local PDF for CPU AMN-NMC byte-model experiments.

The single source is not an independent, representative general-language corpus.
Do not commit the generated training text to a public repository.
"""
import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SOURCE_SHA256 = "bc79c1ffdb3e78327496cff4d7e922d613d60275904a141bcf469e68cf5e1d20"


def prepare_pdf(pdf, output, *, expected_sha256=SOURCE_SHA256):
    import fitz
    pdf = Path(pdf)
    actual_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError("Source PDF SHA-256 mismatch: other training material is not permitted")
    doc = fitz.open(pdf)
    if len(doc) != 80:
        raise ValueError("Expected the inspected 80-page source document")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, excluded, stats = [], [], collections.Counter()
    for ix, page in enumerate(doc):
        number = ix + 1
        if number >= 78:
            excluded.append({"page": number, "reason": "appendix/forms/prompts"})
            continue
        blocks = [b for b in page.get_text("blocks", sort=True)
                  if len(b) >= 7 and b[6] == 0 and isinstance(b[4], str) and b[4].strip()]
        parts, prev_bottom, prev_line = [], None, None
        for block in blocks:
            line = unicodedata.normalize("NFC", block[4].replace("\u00ad", "").strip())
            line = re.sub(r"\s+", " ", line)
            if not line:
                continue
            if line.count("|") >= 2:
                stats["removed_table_rows"] += 1
                continue
            if len(line) < 12 and re.fullmatch(r"[0-9\s.,/%():+\-]+", line):
                stats["removed_table_cell_fragments"] += 1
                continue
            if ("MEB.TT." in line or "2025.028886.01" in line or
                    "671141" in line or "2024.07.05" in line or
                    ("protokol numaralı karar" in line and number == 31)):
                stats["removed_ethics_identifier_lines"] += 1
                continue
            gap = 999 if prev_bottom is None else block[1] - prev_bottom
            heading = bool(re.match(r"^(?:BÖLÜM\s+[IVX]+|\d+(?:\.\d+)*\.\s+\S|ÖZET|EK\s*\d+)", line)) and len(line) < 125
            if parts and (heading or gap > 9):
                parts.append("\n\n" + line)
            elif parts and prev_line and prev_line.endswith(("-", "‐")):
                parts.append(line)
            elif parts:
                parts.append(" " + line)
            else:
                parts.append(line)
            prev_bottom = block[3]
            prev_line = line
        text = "".join(parts).strip()
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Training entries should not begin or end with fragmented page sentences.
        if text and text[0].islower():
            match = re.search(r"[.!?](?:\s|$)", text)
            if match and match.end() < min(len(text) // 2, 450):
                stats["trimmed_incomplete_page_prefixes"] += 1
                text = text[match.end():].lstrip()
        if text and text[-1] not in ".!?" and re.search(r"[.!?](?:\s|$)", text):
            matches = list(re.finditer(r"[.!?](?:\s|$)", text))
            end = matches[-1].end()
            if len(text) - end < min(450, len(text) // 3):
                stats["trimmed_incomplete_page_suffixes"] += 1
                text = text[:end].rstrip()
        if not text or "\ufffd" in text:
            excluded.append({"page": number, "reason": "missing or invalid extracted text"})
            continue
        records.append({"page": number, "text": text,
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    ranges = {"train": [1, 63], "validation": [64, 70], "test": [71, 77]}
    unique = set()
    for split, (start, finish) in ranges.items():
        group = [record for record in records if start <= record["page"] <= finish]
        with (output / (split + ".jsonl")).open("x", encoding="utf-8") as handle:
            for record in group:
                if record["sha256"] in unique:
                    raise ValueError("Repeated page across partitions")
                unique.add(record["sha256"])
                handle.write(json.dumps({"text": record["text"]}, ensure_ascii=False) + "\n")
        stats[split + "_pages"] = len(group)
        stats[split + "_bytes"] = sum(len(record["text"].encode("utf-8")) for record in group)
    manifest = {
        "dataset_kind": "single_academic_pdf_internal_research",
        "source_sha256": actual_sha, "source_pages": 80, "usable_pages": len(records),
        "source_document_count": 1,
        "accepted_for_general_pretraining": False,
        "source_publication_license": "not supplied; do not publish raw text",
        "split_type": "same_document_page_holdout; NOT independent validation",
        "split_page_ranges": ranges,
        "split_files_sha256": {
            name: hashlib.sha256((output / (name + ".jsonl")).read_bytes()).hexdigest()
            for name in ranges
        },
        "exclusions": excluded, "stats": dict(stats),
        "unique_page_hashes": len(unique),
        "total_bytes": sum(len(r["text"].encode("utf-8")) for r in records),
        "privacy": "Do not upload full PDF-derived training text to public GitHub",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Locked-source PDF to private raw byte-LM text")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    print(json.dumps(prepare_pdf(args.pdf, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
