"""No-network regression tests for Turkish world data ingestion and filtering."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from prepare_amn_world_facts_v1 import (
    clean_lead,make_qa,official_names,digest,norm
)


def test_clean_dog_source_and_three_grounded_questions():
    a={"requested":"Köpek","category":"hayvanlar","title":"Köpek",
       "extract":"Köpek, evcilleştirilmiş bir memeli hayvan türüdür ve insanlar ile birlikte yaşayan hayvanlardan biridir. Köpekler farklı boyutlarda olabilir.",
       "url":"https://tr.wikipedia.org/wiki/K%C3%B6pek","revision":10}
    fact,reason=clean_lead(a)
    assert reason=="accepted"
    assert fact.startswith("Köpek,")
    assert fact.endswith("biridir.")
    q=make_qa(a,fact)
    assert len(q)==3
    assert all(item["answer"]==fact for item in q)
    assert q[0]["source_url"].startswith("https://tr.wikipedia.org/")
    assert q[0]["prompt"]=="Köpek nedir?"

def test_reject_markup_disambiguation_and_wrong_title():
    article={"requested":"Köpek","category":"hayvanlar","title":"Köpek",
      "extract":"Köpek, bazı türlerin ortak adıdır ve bu sayfa bir liste oluşturur."}
    assert clean_lead(article)[1]=="disambiguation_or_markup"
    article["extract"]="Bu sıradan açıklama, köpek hakkında değil, farklı bir konuyla ilgili açıklamadır."
    assert clean_lead(article)[1]!="accepted"
    article["failure"]="download_failed"
    assert clean_lead(article)[1]=="download_failed"

def test_official_2025_name_counts_and_unique_prompts():
    names=official_names()
    assert len(names)>20
    keys=[norm(x["prompt"]) for x in names]
    assert len(set(keys))==len(keys)
    final=names[-1]
    assert "20 tane erkek insan 8 tane kadın insan ismi yaz"==final["prompt"]
    answer=final["answer"]
    male=answer.split("Erkek isimleri: ")[1].split(". Kız isimleri: ")[0].split(", ")
    female=answer.split("Kız isimleri: ")[1].rstrip(".").split(", ")
    assert len(male)==20
    assert len(female)==8
    assert all(x["source"]=="TÜİK ADNKS 2025" for x in names)
