"""Six post-training questions against the *saved* v2 AMN checkpoint.

No model weights are trained or edited. Generated answers are archived as-is.
"""
import hashlib
import json
from pathlib import Path

import torch

from amn_open_turkish_sp import load_sp, restore

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"results/open_turkish_retrain_v2/best.pt"
TOKENIZER=ROOT/"results/open_turkish_retrain_v2/tokenizer.model"
OUTPUT=ROOT/"results/open_turkish_retrain_v2/chat_probes"
QUESTIONS=(
    "Bir robot engellerden kaçınmak için hangi sensörleri kullanabilir?",
    "Python dilinde for döngüsü ne işe yarar? Bir örnek verir misin?",
    "Bir elektrik devresinde Ohm kanunu nasıl kullanılır?",
    "Kablosuz ağdaki gecikmenin nedenlerini açıklar mısın?",
    "Yeni başlayan birine yazılım testi neden önemlidir, anlatır mısın?",
    "Bir arkadaşına teşekkür etmek için doğal bir Türkçe cümle yaz.",
)

def main():
    torch.set_num_threads(2)
    sp=load_sp(TOKENIZER)
    model,meta=restore(MODEL)
    model.eval()
    rows=[]
    for i,question in enumerate(QUESTIONS):
        answer=model.generate(
            sp,question,max_new_tokens=96,temperature=0,top_k=20,seed=500+i)
        row={"question":question,**answer}
        rows.append(row)
        print("REAL_MODEL_RESPONSE",json.dumps(row,ensure_ascii=False),flush=True)
    result={
        "architecture":"AMNSubwordLM-v1",
        "checkpoint_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "tokenizer_sha256":hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "best_step":meta.get("best_step"),
        "training_this_run":False,
        "decode":{"temperature":0,"max_new_tokens":96},
        "answers":rows,
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"answers.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Yeniden eğitilen AMN modeline sorulan 6 gerçek soru","",
           "Girdi serbest Türkçe metindir; altı cevap da kaydedilmiş AMN checkpoint'inden",
           "greedy üretim ile gelmiştir. İçeriğe insan veya başka LLM tarafından düzeltme yapılmamıştır.",""]
    for i,row in enumerate(rows,1):
        lines.extend([f"## Soru {i}",row["question"],"",
                      "**AMN cevabı:**",row["text"] or "(boş yanıt)","",
                      f"EOS: {row['eos']}; token: {row['tokens_generated']}",""])
    (OUTPUT/"answers_tr.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":
    main()
