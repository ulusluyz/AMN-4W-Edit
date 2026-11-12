"""Nine exact user questions, answered by SAVED AMN-TM v3.1 replay8 model.

No training, scripted answer templates, web retrieval, or replacement with
another language model. Responses are generated autoregressively from the
checkpoint and copied verbatim to the audit output.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_tm_v31 import restore

MODEL=ROOT/"results/amn_tm_v31_replay8/trained.pt"
TOKENIZER=ROOT/"results/amn_tm_v31/tokenizer.model"
DEST=ROOT/"results/amn_tm_v31_replay8/direct_conversation_9"

QUESTIONS=[
    "Web geliştirme nedir",
    "Teknik yön nedir",
    "Denklem nedir",
    "Köpek nedir",
    "10.000 kuş türü mü vardır",
    "Kadın kime denir",
    "Erkek kime denir",
    "İnsan cinsiyetleri neler",
    "20 tane erkek insan 8 tane kadın insan ismi yaz",
]

def main():
    torch.set_num_threads(2)
    if not MODEL.is_file() or not TOKENIZER.is_file():
        raise FileNotFoundError("Saved AMN replay8 checkpoint or tokenizer absent")
    sp=spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    model,metadata=restore(MODEL)
    model.eval()
    entries=[]
    for i,q in enumerate(QUESTIONS):
        begin=time.perf_counter()
        out=model.generate(sp,q,max_new_tokens=120,temperature=0,
                           top_k=0,seed=1100+i)
        row={
            "number":i+1,"question_exact":q,
            "model_raw_answer":out["text"],
            "generated_tokens":out["tokens"],
            "eos":out["eos"],
            "inference_seconds":round(time.perf_counter()-begin,3),
        }
        entries.append(row)
        print("REAL_AMN_RESPONSE",json.dumps(row,ensure_ascii=False),flush=True)
    audit={
        "source":"direct user questions in current ChatGPT conversation",
        "model":"AMNTMv3.1 replay8",
        "checkpoint_path":str(MODEL.relative_to(ROOT)),
        "checkpoint_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "tokenizer_path":str(TOKENIZER.relative_to(ROOT)),
        "tokenizer_sha256":hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "training_in_this_run":False,
        "retrieval_in_this_run":False,
        "human_edits_to_model_answers":False,
        "decoding":"greedy, temperature=0, max_new_tokens=120",
        "questions":entries,
    }
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/"answers.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=[
        "# AMN-TM v3.1 replay8 — doğrudan konuşma: 9 yeni soru",
        "",
        "Sorular kullanıcının yazdığı biçimde gönderildi. Yanıtlar tamamen",
        "kaydedilmiş AMN replay8 checkpoint'inden çıktı; yeniden eğitim yok,",
        "cevap şablonu yok, başka LLM ile düzeltme yok.",
        "Greedy decoding, max 120 yeni token. Yanıtlar doğruluk garantisi taşımaz.",
        "",
    ]
    for x in entries:
        lines.extend([
            "## Soru "+str(x["number"]),
            "**Soru:** "+x["question_exact"],
            "",
            "**AMN cevabı (ham):**",
            x["model_raw_answer"] or "(boş cevap)",
            "",
            "Üretilen token: "+str(x["generated_tokens"])+
            " / EOS: "+str(x["eos"]),
            "",
        ])
    (DEST/"answers_tr.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":
    main()
