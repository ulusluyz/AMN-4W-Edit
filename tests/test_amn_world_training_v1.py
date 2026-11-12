"""Regression tests for curated Wikipedia/TUIK real-knowledge AMN training."""
import sys,json,hashlib
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))

from train_amn_world_v1 import core_sample
from prepare_amn_world_facts_v1 import clean_lead,MANUAL_EXCLUSIONS

def test_manually_audited_wrong_sense_facts_are_rejected():
    assert "Batarya" in MANUAL_EXCLUSIONS
    assert "Afrika" in MANUAL_EXCLUSIONS
    assert "Matris" in MANUAL_EXCLUSIONS
    assert "Virüs" in MANUAL_EXCLUSIONS
    for name in ("Batarya","Afrika","Matris","Virüs","Donanım"):
        item={"requested":name,"title":name,"category":"sanity",
              "extract":name+", bu terimin tanımı olarak kullanılabilecek eksiksiz bir örnek cümledir."}
        fact,reason=clean_lead(item)
        assert fact is None and reason=="manual_review_reject"


def test_important_world_concepts_are_not_missing_from_core_rehearsal():
    assert core_sample({"category":"hayvanlar","topic":"Köpek"})
    assert core_sample({"category":"matematik","topic":"Denklem"})
    assert core_sample({"category":"resmi_isimler","topic":"2025 resmi isimler iki grup"})
    assert core_sample({"category":"uzman_kaynakli_temel_kavram","topic":"Web geliştirme"})
    assert not core_sample({"category":"cografya","topic":"İtalya"})


def test_training_manifest_hashes_and_source_partitions_if_materialized():
    root=ROOT/"data/amn_world_v1"
    if not (root/"manifest.json").exists():
        pytest.skip("The curated data archive is materialized by GitHub Actions at training runtime")
    from train_amn_world_v1 import validate_data
    manifest,audit,splits=validate_data(root)
    assert manifest["source_file_sha256"]==audit["source_sha256"]
    assert len(splits["train"])>=200
    assert len(splits["test"])>=10
    assert audit["manually_rejected_count"]>=35
