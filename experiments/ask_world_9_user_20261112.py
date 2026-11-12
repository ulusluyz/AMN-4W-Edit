"""Nine exact user questions against frozen AMN v3.1 WORLD checkpoint.
No training or lookup; greedy autoregressive unedited output.
"""
import hashlib,json,sys,time
from pathlib import Path
import torch
import sentencepiece as spm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_tm_v31 import restore
MODEL=ROOT/"results/amn_world_v1_train/best.pt"
TOKENIZER=ROOT/"results/amn_tm_v31/tokenizer.model"
OUTPUT=ROOT/"results/amn_world_v1_train/user_questions_20261112"
QUESTIONS=[
"Cumhuriyet nedir",
"Kedi nedir",
"Güneş nedir",
"Top nasıl oynanır",
"Oy pusuladı nedir",
"Salata nasıl yapılır",
"Saksı nedir",
"Saat nasıl kullanılır",
"Pusula nedir"
]
def main():
    torch.set_num_threads(2)
    model,meta=restore(MODEL)
    sp=spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    model.eval()
    answers=[]
    for i,q in enumerate(QUESTIONS,1):
        start=time.perf_counter()
        out=model.generate(sp,q,max_new_tokens=120,temperature=0,top_k=0,seed=1200+i)
        item={"number":i,"question":q,"raw_answer":out["text"],"tokens":out["tokens"],"eos":out["eos"],
              "inference_seconds":round(time.perf_counter()-start,4)}
        answers.append(item)
        print("AMN_RAW",json.dumps(item,ensure_ascii=False),flush=True)
    audit={
        "model":"AMNTMv3.1","checkpoint":"results/amn_world_v1_train/best.pt",
        "checkpoint_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "tokenizer_sha256":hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "new_training":False,"retrieval":False,"edited_answers":False,
        "temperature":0,"max_new_tokens":120,"questions":answers
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"answers.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    text=["# AMN v3.1 dünya bilgisi — kullanıcının 9 sorusu","",
      "Dondurulmuş model checkpoint'inden üretilen gerçek cevaplar.",
      "Eğitim veya düzenleme yapılmadı; temperature=0.", ""]
    for a in answers:
        text.extend(["## "+str(a["number"])+". "+a["question"],
            a["raw_answer"] or "(boş cevap)",""])
    (OUTPUT/"answers_tr.md").write_text("\n".join(text),encoding="utf-8")
if __name__=="__main__":main()
