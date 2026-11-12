"""Test whether rehearsed AMN answers are truly conditioned on their questions.

For each true Q/A pair, contrast teacher-forced NLL of its answer under the
correct prompt against the same answer under an unrelated question. This
prevents calling unconditional Turkish sentence memorization "Q/A learning".
Original checkpoint and repaired checkpoint are both evaluated unmodified.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

import torch
import sentencepiece as spm
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_tm_v3 import build_sft,collate_lm,IGNORE
from amn_tm_v31 import restore
from fix_amn_v31_rehearsal import REHEARSAL_INDICES

@torch.inference_mode()
def answer_nll(model,sp,question,answer):
    row=build_sft(sp,question,answer,max_len=192)
    if row is None:raise ValueError("Question/answer exceeds 192 tokens")
    x,y=collate_lm([row],sp.pad_id())
    logits=model(x)
    nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                        y.reshape(-1),ignore_index=IGNORE,reduction="mean")
    return float(nll)

def test_conditioning(model,sp,items):
    model.eval()
    rows=[]
    for i,r in enumerate(items):
        wrong=items[(i+1)%len(items)]
        true_nll=answer_nll(model,sp,r["question"],r["training_reference"])
        mismatched_nll=answer_nll(model,sp,wrong["question"],r["training_reference"])
        rows.append({
          "correct_question":r["question"],
          "wrong_question":wrong["question"],
          "target_answer":r["training_reference"],
          "true_prompt_nll":true_nll,
          "swapped_prompt_nll":mismatched_nll,
          "margin_wrong_minus_true":mismatched_nll-true_nll,
          "correct_prompt_preferred":true_nll<mismatched_nll
        })
    return {"n":len(rows),
            "positive_margin_count":sum(x["correct_prompt_preferred"] for x in rows),
            "mean_margin_wrong_minus_true":sum(x["margin_wrong_minus_true"] for x in rows)/len(rows),
            "per_question":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="results/amn_tm_v31")
    p.add_argument("--repaired",default="results/amn_tm_v31_rehearsal")
    p.add_argument("--audit",default="results/amn_tm_v31/seen_training_questions/audit.json")
    a=p.parse_args()
    b=Path(a.base);r=Path(a.repaired)
    audit=json.loads(Path(a.audit).read_text(encoding="utf-8"))
    items=[audit["results"][i] for i in REHEARSAL_INDICES]
    sp=spm.SentencePieceProcessor(model_file=str(b/"tokenizer.model"))
    torch.set_num_threads(2)
    original,_=restore(b/"best.pt")
    updated,_=restore(r/"trained.pt")
    result={
        "scope":"6 genuine previously seen training examples",
        "purpose":"Verify whether answers depend on their own question instead of generic text",
        "before":test_conditioning(original,sp,items),
        "after":test_conditioning(updated,sp,items),
        "claim_limit":"This only measures question-conditioned reference likelihood on rehearsed examples; not human judgment or generalization."
    }
    (r/"question_conditioning.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("QUESTION_CONDITIONING",json.dumps(
        {"before":{k:v for k,v in result["before"].items() if k!="per_question"},
         "after":{k:v for k,v in result["after"].items() if k!="per_question"}},
         ensure_ascii=False),flush=True)

if __name__=="__main__":main()
