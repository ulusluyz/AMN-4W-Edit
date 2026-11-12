"""PDF-bound data pipeline: source-locking, privacy, partitions and guard checks."""
import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amn_nmc.pdf_corpus import prepare_pdf, SOURCE_SHA256
from amn_nmc import train_pdf


def synthetic_pdf(path):
    import fitz
    doc = fitz.open()
    for index in range(80):
        page = doc.new_page()
        text = ("Turkce metin bu teknik testte bir metin dizisidir. "
                "Bu cumle model girdisini denetler. "
                + f"Test sayfasi {index + 1}. ") * 9
        assert page.insert_textbox(fitz.Rect(50, 80, 540, 780), text, fontsize=10) >= 0
    doc.save(path)
    doc.close()


def test_source_lock_refuses_unapproved_pdf(tmp_path):
    pdf = tmp_path / "other.pdf"
    synthetic_pdf(pdf)
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_pdf(pdf, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_pdf_single_source_partition_integrity(tmp_path, monkeypatch):
    pdf = tmp_path / "synthetic.pdf"
    synthetic_pdf(pdf)
    source_sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    corpus = tmp_path / "only_this_pdf"
    manifest = prepare_pdf(pdf, corpus, expected_sha256=source_sha)
    assert manifest["source_document_count"] == 1
    assert manifest["accepted_for_general_pretraining"] is False
    assert manifest["usable_pages"] == 77
    assert manifest["stats"]["train_pages"] == 63
    assert manifest["stats"]["validation_pages"] == 7
    assert manifest["stats"]["test_pages"] == 7
    for split in ("train", "validation", "test"):
        actual = hashlib.sha256((corpus / (split + ".jsonl")).read_bytes()).hexdigest()
        assert actual == manifest["split_files_sha256"][split]
    monkeypatch.setattr(train_pdf, "SOURCE_SHA256", source_sha)
    assert train_pdf.validate_source(corpus)["source_sha256"] == source_sha
    with (corpus / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"text":"extra unverified data"}\n')
    with pytest.raises(ValueError, match="Modified training partition"):
        train_pdf.validate_source(corpus)


def test_no_legacy_wikipedia_source_is_accepted(tmp_path):
    corpus = tmp_path / "fake_wikipedia"
    corpus.mkdir()
    manifest = {"dataset_kind": "Wikipedia", "source_sha256": SOURCE_SHA256}
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not the explicitly approved PDF"):
        train_pdf.validate_source(corpus)
