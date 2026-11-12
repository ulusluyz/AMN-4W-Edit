"""Ask additional Turkish questions of the frozen native AMN checkpoint.

Runs *inference only*. It never trains or rewrites weights.
Every response is produced from the saved model, not drafted by an LLM.
"""
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from amn_open_turkish import load_checkpoint

QUESTIONS = (
    "Bir robotun engellere çarpmadan hareket etmesi için hangi sensörler kullanılabilir?",
    "Python dilinde bir döngü ne işe yarar? Kısa bir örnekle açıkla.",
    "Elektrik devresinde direnç ve akım arasında nasıl bir ilişki vardır?",
    "Kablosuz ağda bağlantı kalitesi neden düşebilir?",
    "Türkçe dilinde bir arkadaşına teşekkür etmek için kısa ve doğal bir cümle yaz.",
    "Bir yazılım projesini test ederken hangi hatalara dikkat etmeliyiz?",
)

def run():
    path = ROOT / "results/open_turkish/best.pt"
    data = path.read_bytes()
    model, metadata = load_checkpoint(path)
    model.eval()
    outputs = []
    for i, question in enumerate(QUESTIONS):
        response = model.complete(
            question,
            max_new_bytes=320,
            temperature=0.7,
            top_k=12,
            seed=20261112 + i,
        )
        item = {"question":question, **response}
        outputs.append(item)
        print("QUESTION_AND_REAL_MODEL_ANSWER", json.dumps(item, ensure_ascii=False), flush=True)
    payload = {
        "model":"AMNOpenTurkish-v1",
        "checkpoint_sha256":hashlib.sha256(data).hexdigest(),
        "checkpoint_best_step":metadata.get("step"),
        "inference_only":True,
        "decode":{"max_new_bytes":320, "temperature":0.7, "top_k":12},
        "results":outputs,
    }
    folder=ROOT/"results/open_turkish/more_questions"
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"answers.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines=["# AMN modeline sorulan 6 yeni Türkçe soru","",
           "Kayıtlı model ağırlıklarından üretilmiş ham yanıtlar; düzenlenmedi.",
           "Bu deneyde ek eğitim yapılmadı.",""]
    for i,row in enumerate(outputs,1):
        lines.extend([f"## {i}. soru",f"**Soru:** {row['question']}","",
                      f"**Modelin gerçek yanıtı:** {row['text'] or '(boş yanıt)'}","",
                      f"UTF-8 geçerli: {row['valid_utf8']}; EOS: {row['eos']}",""])
    (folder/"answers_tr.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    run()
