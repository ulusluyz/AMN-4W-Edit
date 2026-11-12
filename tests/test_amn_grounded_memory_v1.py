"""Source-only AMN memory tests: correct subject, no leaked test facts and no
"Web..." answer when the question does not mention that topic.
"""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_grounded_memory_v1 import MemoryIndex,NOT_FOUND,normalize

DATA=ROOT/"data/amn_world_v1"

def test_turkish_normalization_and_exact_training_reference():
    assert normalize("KEDİ NEDİR?!")=="kedi nedir"
    assert normalize("İNSAN")=="insan"
    assert normalize("IŞIK")=="ışık"
    memory=MemoryIndex(DATA/"train.jsonl")
    for name in ("Kedi","Köpek","Denklem","İnsan","Erkek","Kadın"):
        response=memory.answer(name+" nedir?")
        assert response.found
        assert response.topic==name
        assert response.source_url.startswith("https://")
        assert response.answer
        assert response.match_type in ("exact_train_question","exact_topic")
    assert "Felis catus" in memory.answer("Kedi nedir?").answer
    assert "eşitliğini" in memory.answer("Denklem nedir?").answer

def test_no_unrelated_topic_substitution_when_question_unknown():
    memory=MemoryIndex(DATA/"train.jsonl")
    for question in ("Kedi nasıl kullanılır","Salata nasıl yapılır",
                     "Top nasıl oynanır","Bilinmeyen nesne nedir",
                     "Oy pusuladı nedir","Oyun saatleri nerede",
                     "Denklem nasıl çözülür","10.000 kuş türü mü vardır"):
        result=memory.answer(question)
        assert result.answer==NOT_FOUND
        assert not result.found
        assert result.source_url==""
    assert "Web" not in memory.answer("Kedi nasıl kullanılır").answer

def test_train_only_memory_no_validation_test_leakage():
    memory=MemoryIndex(DATA/"train.jsonl")
    for split in ("val","test"):
        rows=[json.loads(s) for s in (DATA/(split+".jsonl")).read_text(encoding="utf-8").splitlines() if s]
        topics={r["topic"] for r in rows if r["source"]=="Turkish Wikipedia"}
        for topic in sorted(topics):
            assert not memory.answer(topic+" nedir?").found,topic

def test_names_preserve_actual_source_not_fictional_list():
    memory=MemoryIndex(DATA/"train.jsonl")
    reply=memory.answer("20 tane erkek insan 8 tane kadın insan ismi yaz")
    assert reply.found and reply.match_type=="exact_train_question"
    assert reply.source=="TÜİK ADNKS 2025"
    assert "Alparslan" in reply.answer
    assert "Alya" in reply.answer

def test_topic_routing_does_not_use_substrings_or_wildcard_match():
    memory=MemoryIndex(DATA/"train.jsonl")
    assert memory.answer("Kedi nedir").topic=="Kedi"
    assert not memory.answer("Kediler neden miyavlar?").found
    assert not memory.answer("Güneş ve köpek arasındaki fark nedir?").found
