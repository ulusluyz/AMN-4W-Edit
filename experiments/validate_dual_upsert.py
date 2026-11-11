"""Evaluate untouched Dual AMN weights versus Upsert under identical probes.

RUN: python experiments/validate_dual_upsert.py
No training occurs. Output is archived by GitHub Actions.
"""
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from relational_qa import (
    FACTS, ID, VOCAB, VALUES, DualAddressAMN, AssociativeAMN, generate
)
from dual_upsert import DualAddressUpsertAMN, last_write_mask
from validate_three_models_turkish import (
    make_last_write_batch, render_fact, render_question
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "results/relational_qa/equal50"
OUTPUT = ROOT / "results/dual_upsert"


def load(kind, seed):
    cls = {
        "dual_original": DualAddressAMN,
        "dual_upsert": DualAddressUpsertAMN,
        "amn_original": AssociativeAMN,
    }[kind]
    model = cls(48)
    source = "amn" if kind == "amn_original" else "dual"
    record = torch.load(
        CHECKPOINTS / f"{source}_{seed}.pt",
        weights_only=True, map_location="cpu"
    )
    model.load_state_dict(record["state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def check_batch(models, facts, query, correct):
    return {
        kind: {"correct": int((model(facts, query).argmax(-1) == correct).sum()),
               "total":len(correct)}
        for kind, model in models.items()
    }


@torch.no_grad()
def evaluate(seed):
    models = {kind:load(kind, seed) for kind in (
        "dual_original","dual_upsert","amn_original")}
    original, upsert = models["dual_original"], models["dual_upsert"]
    clean = {kind:{"correct":0,"total":0} for kind in models}
    max_delta = 0.0
    for nf, total in ((4,3000),(6,1000),(8,1000),(12,1000)):
        rng = random.Random(seed + (300000 if nf == 4 else 80000))
        # Exactly match the earlier 3,000 held-out sequence for 4 facts.
        count = 0
        scores = {kind:0 for kind in models}
        while count < total:
            batch = min(128,total-count)
            x, q, y = generate("test",batch,nf,rng)
            assert bool(last_write_mask(x).all()), "Clean test accidentally contains repeated keys"
            l0=original(x,q)
            l1=upsert(x,q)
            diff=float((l0-l1).abs().max())
            max_delta=max(max_delta,diff)
            # Output-identical for every unique-key input, not just accuracy.
            assert torch.equal(l0,l1), (seed,nf,diff)
            for kind,m in models.items():
                pred=(l0 if kind=="dual_original" else
                      l1 if kind=="dual_upsert" else m(x,q))
                scores[kind]+=int((pred.argmax(-1)==y).sum())
            count+=batch
        clean[nf]={kind:{"correct":score,"total":total} for kind,score in scores.items()}
    x,q,y,demos=make_last_write_batch(seed, count=500)
    # The target key occurs exactly twice. Distractor keys MAY also repeat
    # with different values; that is a valid additional upsert stress case.
    target_occurrences = (x[:, :, :2] == q[:, None, :]).all(dim=-1).sum(dim=-1)
    assert bool((target_occurrences == 2).all()), "Expected two target writes per episode"
    assert bool(((~last_write_mask(x)).sum(dim=-1) >= 1).all())
    last_write=check_batch(models,x,q,y)
    examples = []
    if seed==0:
        logits={kind:m(x[:len(demos)],q[:len(demos)]) for kind,m in models.items()}
        for i in range(len(demos)):
            old_ok=int(logits["dual_original"][i].argmax())==int(y[i])
            new_ok=int(logits["dual_upsert"][i].argmax())==int(y[i])
            if not old_ok and new_ok and len(examples)<4:
                row=dict(demos[i])
                row["responses"]={
                    kind:VOCAB[int(value[i].argmax())]
                    for kind,value in logits.items()
                }
                examples.append(row)
    return {"seed":seed,"unchanged_logits_max_abs":max_delta,
            "clean_by_facts":clean,"last_write":last_write,
            "update_examples":examples}


def main():
    torch.set_num_threads(1)
    records=[]
    t=time.perf_counter()
    for seed in range(5):
        r=evaluate(seed)
        records.append(r)
        print("SEED",seed,"UPDATES",r["last_write"],
              "CLEAN",r["clean_by_facts"],flush=True)
    aggregated = {}
    for k in ("amn_original","dual_original","dual_upsert"):
        updates = sum(row["last_write"][k]["correct"] for row in records)
        clean = {nf:sum(row["clean_by_facts"][nf][k]["correct"] for row in records)
                 for nf in (4,6,8,12)}
        aggregated[k] = {"last_write_correct":updates,"last_write_total":2500,
                         "clean_correct_by_facts":clean,
                         "clean_total_by_facts":{4:15000,6:5000,8:5000,12:5000}}
    # No-loss acceptance criteria: original Dual scores retained on every
    # clean item exactly; exact logits unchanged; 100% last-write is required.
    for nf in (4,6,8,12):
        assert aggregated["dual_original"]["clean_correct_by_facts"][nf] == aggregated["dual_upsert"]["clean_correct_by_facts"][nf]
    assert all(row["unchanged_logits_max_abs"]==0.0 for row in records)
    assert aggregated["dual_upsert"]["last_write_correct"] == 2500, (
        "Upsert failed to reach 100% on the update task")
    result={
        "protocol":{
            "checkpoint_training_steps":50, "training_this_run":False,
            "seeds":[0,1,2,3,4],"source":"AMN-4W-Edit",
            "test_triples":len(FACTS["test"]),
            "data_split":"same hashed held-out triples as earlier",
            "last_write_policy":"exact entity+relation ID; latest occurrence wins",
            "known_boundary":"structured facts; no free-form Turkish parsing",
        },
        "aggregate":aggregated,"seeds":records
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"validation.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Dual AMN upsert — Türkçe model çıktıları","",
           "**Uyarı:** Türkçe cümleler sabit bir biçimlendiriciden gelir.",
           "Modeller yalnız tek kelimelik cevap tahmin eder; serbest Türkçe anlamaz.",""]
    for i,row in enumerate(records[0]["update_examples"],1):
        lines.extend([f"## Örnek {i}",
                      " ".join(row["facts"]),
                      "**Soru:** "+row["question"],
                      "**Beklenen son değer:** "+row["latest_write_gold"],
                      "- Standart AMN: **"+row["responses"]["amn_original"]+"**",
                      "- Eski Dual AMN: **"+row["responses"]["dual_original"]+"**",
                      "- Upsert Dual AMN: **"+row["responses"]["dual_upsert"]+"**",""])
    lines.extend(["## Kümülatif başarı","",
                  "| Model | Güncelleme | 4 kayıt | 6 kayıt | 8 kayıt | 12 kayıt |",
                  "|---|---:|---:|---:|---:|---:|"])
    for k in ("amn_original","dual_original","dual_upsert"):
        a=aggregated[k]
        lines.append("| "+k+" | "+str(a["last_write_correct"])+"/2500 | "+" | ".join(
            f"{a['clean_correct_by_facts'][n]}/{a['clean_total_by_facts'][n]}" for n in (4,6,8,12))+" |")
    lines.extend(["","**Aynı ağırlıklar:** Dual upsert yeni parametre eklemedi; "
                  "sadece tekrarlanan anahtarların eski kayıtlarını Gram çözümünden çıkardı.",
                  "Tekrarlanan anahtar bulunmadığında modelin logitleri bit düzeyinde aynıdır.",""])
    (OUTPUT/"outputs_tr.md").write_text("\n".join(lines),encoding="utf-8")
    print("PASS",json.dumps(aggregated,ensure_ascii=False),flush=True)
    print("VALIDATION_SECONDS",round(time.perf_counter()-t,3),flush=True)


if __name__=="__main__":
    main()
