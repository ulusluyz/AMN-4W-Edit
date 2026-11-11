"""Evaluate the three *existing* 50-step models under identical conditions.

Outputs actual Turkish-formatted examples by applying a COMMON, explicitly
rule-based parser to prose. The models themselves accept structured fields,
not unstructured language. No model is retrained in this file.

Run from repo root:
 python experiments/validate_three_models_turkish.py
"""
import json
import random
import re
from pathlib import Path

import torch
from torch.nn import functional as F

from relational_qa import (
    NAMES, RELS, VALUES, FACTS, ID, VOCAB, AssociativeAMN,
    DualAddressAMN, TransformerQA, generate, split_of
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "results/relational_qa/equal50"
OUT = ROOT / "results/real_turkish_3models"

GENITIVE = {
    "Ali": "Ali'nin", "Ayşe": "Ayşe'nin", "Can": "Can'ın",
    "Deniz": "Deniz'in", "Ece": "Ece'nin", "Efe": "Efe'nin",
    "Elif": "Elif'in", "Emre": "Emre'nin", "Mert": "Mert'in",
    "Mina": "Mina'nın", "Murat": "Murat'ın", "Nisa": "Nisa'nın",
    "Okan": "Okan'ın", "Selin": "Selin'in", "Sena": "Sena'nın",
    "Umut": "Umut'un", "Yaren": "Yaren'in", "Yiğit": "Yiğit'in",
    "Zeynep": "Zeynep'in", "Arda": "Arda'nın",
}
LOCATION_FORM = {
    "ev": "evde", "okul": "okulda", "park": "parkta",
    "ofis": "ofiste", "bahçe": "bahçede", "kafe": "kafede",
    "salon": "salonda", "garaj": "garajda",
}
REV_LOCATION = {v: k for k, v in LOCATION_FORM.items()}
REV_GENITIVE = {v: k for k, v in GENITIVE.items()}

assert len(GENITIVE) == len(NAMES)
assert all(name in GENITIVE for name in NAMES)


def render_fact(triple):
    name, rel, value = triple
    if rel == "renk":
        return f"{GENITIVE[name]} kaleminin rengi {value}."
    if rel == "konum":
        return f"{name} şu anda {LOCATION_FORM[value]}."
    raise ValueError("Unsupported relation")


def render_question(pair):
    name, rel = pair
    if rel == "renk":
        return f"{GENITIVE[name]} kaleminin rengi nedir?"
    if rel == "konum":
        return f"{name} şu anda nerede?"
    raise ValueError("Unsupported query relation")


def parse_fact(sentence):
    sentence = sentence.strip()
    if not sentence.endswith("."):
        raise ValueError("Only complete supported Turkish sentences accepted")
    stem = sentence[:-1]
    for genitive, name in REV_GENITIVE.items():
        pref = genitive + " kaleminin rengi "
        if stem.startswith(pref):
            value = stem[len(pref):]
            if value not in VALUES["renk"]:
                raise ValueError("Unknown color word: " + value)
            return (name, "renk", value)
    for name in NAMES:
        pref = name + " şu anda "
        if stem.startswith(pref):
            loc = stem[len(pref):]
            if loc not in REV_LOCATION:
                raise ValueError("Unknown location form: " + loc)
            return (name, "konum", REV_LOCATION[loc])
    raise ValueError("Free Turkish sentence unsupported by fixed parser: " + sentence)


def parse_question(question):
    for genitive, name in REV_GENITIVE.items():
        if question == genitive + " kaleminin rengi nedir?":
            return (name, "renk")
    for name in NAMES:
        if question == name + " şu anda nerede?":
            return (name, "konum")
    raise ValueError("Free Turkish question unsupported by fixed parser: " + question)


def parse_text(facts, question):
    triples = [parse_fact(s) for s in facts]
    pair = parse_question(question)
    return (
        torch.tensor([[ID[x] for x in t] for t in triples], dtype=torch.long),
        torch.tensor([ID[x] for x in pair], dtype=torch.long)
    )


def trained_models(seed):
    for kind in ("amn", "dual", "transformer"):
        model = (AssociativeAMN(48) if kind == "amn" else
                 DualAddressAMN(48) if kind == "dual" else TransformerQA(48, 4))
        path = WEIGHTS / f"{kind}_{seed}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"], strict=True)
        model.eval()
        yield kind, model


@torch.no_grad()
def score(model, facts, query, labels):
    logits = model(facts, query)
    return {
        "correct": int((logits.argmax(-1) == labels).sum()),
        "total": len(labels),
        "loss": float(F.cross_entropy(logits, labels))
    }


def make_last_write_batch(seed, count=500):
    """Untrained task: old and new values for the same (entity,relation).

    Both models receive the exact same context and question. Gold answer is
    determined externally for evaluation, NOT supplied as an inference input.
    """
    rng = random.Random(500_000 + seed)
    pool = FACTS["test"]
    values_by_rel = VALUES
    contexts, queries, targets, demos = [], [], [], []
    for _ in range(count):
        desired = rng.choice(pool)
        name, rel, value = desired
        oldval = rng.choice([v for v in values_by_rel[rel] if v != value])
        old = (name, rel, oldval)
        others = rng.sample([f for f in pool if (f[0], f[1]) != (name,rel)], 2)
        latest = rng.choice([old,desired])
        earliest = desired if latest == old else old
        facts = [earliest] + others + [latest]
        queries.append([ID[name], ID[rel]])
        contexts.append([[ID[w] for w in fact] for fact in facts])
        targets.append(ID[latest[2]])
        if len(demos) < 3:
            demos.append({
                "facts": [render_fact(fact) for fact in facts],
                "question": render_question((name, rel)),
                "latest_write_gold": latest[2]
            })
    return (torch.tensor(contexts), torch.tensor(queries),
            torch.tensor(targets), demos)


def render_examples(models, count=12):
    rng = random.Random(20261111)
    raw_facts, raw_query, labels = generate("test", count, 4, rng)
    examples = []
    for i in range(count):
        triples = [[VOCAB[t] for t in fact] for fact in raw_facts[i].tolist()]
        query = [VOCAB[t] for t in raw_query[i].tolist()]
        sentences = [render_fact(t) for t in triples]
        question = render_question(query)
        reconstructed, parsed_q = parse_text(sentences, question)
        assert torch.equal(reconstructed, raw_facts[i])
        assert torch.equal(parsed_q, raw_query[i])
        example = {
            "context": " ".join(sentences),
            "question": question,
            "expected": VOCAB[int(labels[i])],
            "model_answers": {}
        }
        for kind, model in models.items():
            with torch.no_grad():
                logits = model(reconstructed[None], parsed_q[None])
                probs = logits.softmax(-1)[0]
                argmax = int(probs.argmax())
                example["model_answers"][kind] = {
                    "answer": VOCAB[argmax],
                    "confidence": round(float(probs[argmax]), 6),
                    "exact": argmax == int(labels[i]),
                }
        examples.append(example)
    return examples


def main():
    torch.set_num_threads(1)
    assert len(FACTS["test"]) == 31
    # The parser is deliberately NOT a free-Turkish language model.
    known = ["Ali'nin kaleminin rengi mavi.", "Can şu anda parkta."]
    parsed, q = parse_text(known, "Ali'nin kaleminin rengi nedir?")
    assert parsed.tolist() == [[ID["Ali"],ID["renk"],ID["mavi"]],
                               [ID["Can"],ID["konum"],ID["park"]]]
    assert q.tolist() == [ID["Ali"],ID["renk"]]
    unsupported = {}
    for label, text in (
        ("new_word", "Bora'nın kaleminin rengi mavi."),
        ("free_prose", "Ayşe bugün hava güzel olduğu için dışarı çıktı."),
        ("deletion", "Ali'nin kaleminin rengini unut."),
    ):
        try:
            parse_fact(text)
            unsupported[label] = "accepted"
        except ValueError:
            unsupported[label] = "unsupported"
    assert all(x == "unsupported" for x in unsupported.values())
    out = {
        "protocol": {
            "checkpoint_training_steps": 50, "training_examples": 264,
            "dev_triples":25, "heldout_triples":31,
            "test_episodes_per_seed": 3000,
            "overwrite_episodes_per_seed":500,
            "seeds":[0,1,2,3,4],
            "free_turkish_input": False,
            "turkish_text_layer": "handwritten finite grammar parser, identical for all",
            "answer_generation": "single-vocabulary-token classification, not prose generation",
            "novel_words": "unsupported",
            "deletion_operation": "unsupported",
            "multi_hop": "unsupported",
        },
        "unsupported_probes":unsupported, "seeds": [], "examples":[],
    }
    for seed in range(5):
        models = dict(trained_models(seed))
        rng = random.Random(seed+300000)
        clean_stats = {kind:{"correct":0,"total":0,"loss_sum":0.0} for kind in models}
        for offset in range(0,3000,128):
            n = min(128, 3000-offset)
            facts, query, labels = generate("test",n,4,rng)
            for kind, model in models.items():
                res = score(model,facts,query,labels)
                row = clean_stats[kind]
                row["correct"] += res["correct"]
                row["total"] += res["total"]
                row["loss_sum"] += res["loss"]*n
        latest_facts, latest_query, latest_labels, _ = make_last_write_batch(seed)
        update_stats = {}
        for kind, model in models.items():
            result = score(model,latest_facts,latest_query,latest_labels)
            update_stats[kind] = result
            clean_stats[kind]["loss"] = clean_stats[kind].pop("loss_sum")/3000
        if seed == 0:
            out["examples"] = render_examples(models,12)
        out["seeds"].append({"seed":seed,"heldout":clean_stats,"last_write_stress":update_stats})
        print("SEED",seed,"HELDOUT",{
            kind: r["correct"] for kind,r in clean_stats.items()
        },"LAST_WRITE",{
            kind: r["correct"] for kind,r in update_stats.items()
        },flush=True)

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"validation.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    lines = [
        "# Üç eğitilmiş modelin Türkçe çıktıları — gerçek ağırlıklardan",
        "",
        "**UYARI:** Türkçe metinler ortak kural tabanlı ayrıştırıcıyla "
        "kişi–ilişki–değer üçlülerine dönüştürülmüştür. Modeller serbest "
        "Türkçe okumamıştır. Altın cevaplar yalnızca değerlendirme içindir.",
        "Modellerin ürettiği **yalnızca tek kelimelik cevap tokenlarıdır**; "
        "bağlam ve soru cümleleri dış arabirim tarafından biçimlendirilmiştir.",
        ""
    ]
    for i,item in enumerate(out["examples"],1):
        lines.extend([f"## Örnek {i}",item["context"],"",
                      "**Soru:** "+item["question"],
                      "**Gerçek cevap:** "+item["expected"]])
        for k in ("amn","dual","transformer"):
            row=item["model_answers"][k]
            status = "DOĞRU" if row["exact"] else "YANLIŞ"
            lines.append(f"- {k}: **{row['answer']}** ({status}; olasılık {row['confidence']:.4f})")
        lines.append("")
    lines.append("## Sayısal sonuçlar")
    for kind in ("amn","dual","transformer"):
        ok=sum(row["heldout"][kind]["correct"] for row in out["seeds"])
        upd=sum(row["last_write_stress"][kind]["correct"] for row in out["seeds"])
        lines.append(f"- {kind}: 4 kayıtlı held-out **{ok}/15000**, "
                     f"tekrarlanan anahtar güncelleme stresi **{upd}/2500**.")
    lines.extend(["","**Serbest Türkçe / yeni sözcük / silme / iki-hop:** "
                  "Bu checkpointlerde desteklenmiyor. Desteklenmeyen görevde "
                  "başarı oranı uydurulmamıştır.",""])
    (OUT/"outputs_tr.md").write_text("\n".join(lines),encoding="utf-8")
    print("WROTE",OUT/"validation.json",OUT/"outputs_tr.md",flush=True)


if __name__ == "__main__":
    main()
