"""AMN v3.1 controlled learning repair: repeat genuinely seen source Q/A examples.

Not a fake lookup table: all gradient updates go through the original AMN,
GRU, SentencePiece and autoregressive decoder. Provenance is checked against
the original training split and independently replayed SGD audit.

GOAL: distinguish a broken optimization / inference pipeline from the fact
that most of the 17,170 examples were shown zero times in the first 1,420
step run. Hold two seen questions out of this new rehearsal, and compare
before/after both training and control questions.

No claims of open-ended intelligence or heldout generalization.
"""
from __future__ import annotations
import argparse,hashlib,json,random,sys,time,math
from pathlib import Path
import torch
from torch.nn import functional as F
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_tm_v3 import build_sft,collate_lm,IGNORE
from amn_tm_v31 import restore,save
from train_amn_tm_v3 import read_split,prepare_rows

# Immutable indices from 12 verified training examples, picked BEFORE new
# training/inference. Excludes known wrong integral and malformed prompts.
REHEARSAL_INDICES=(2,4,7,8,10,11)
CONTROL_INDICES=(5,9)

def normalize(s):
    return " ".join(s.casefold().split())

def char_f1(hyp,reference):
    # Char multiset overlap: proxy for reference resemblance, NOT semantic accuracy.
    from collections import Counter
    h=Counter(hyp.casefold());r=Counter(reference.casefold())
    overlap=sum((h&r).values())
    return 2*overlap/max(1,len(hyp)+len(reference))

def verify_original(data,model_dir,audit_file):
    metrics=json.loads((model_dir/"metrics.json").read_text(encoding="utf-8"))
    audit=json.loads(audit_file.read_text(encoding="utf-8"))
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    if not (manifest==metrics["data"]==audit["source"]):
        raise RuntimeError("Source training manifest mismatch")
    sha=hashlib.sha256((model_dir/"tokenizer.model").read_bytes()).hexdigest()
    if sha!=metrics["tokenizer_sha256"] or sha!=audit["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer checksum mismatch")
    if not audit.get("original_training_sampling_replayed"):
        raise RuntimeError("Original gradient provenance was not verified")
    if len(audit["results"])<12:
        raise RuntimeError("Incomplete original sample audit")
    sp=spm.SentencePieceProcessor(model_file=str(model_dir/"tokenizer.model"))
    train=prepare_rows(sp,read_split(data,"train"),metrics["effective_training_context_max"])
    by_question={r["prompt"]:r for r in train}
    for rec in audit["results"]:
        if rec["sft_exposures"]<=0:
            raise RuntimeError("Not truly seen by the original SFT updates")
        actual=by_question.get(rec["question"])
        if actual is None or actual["answer"]!=rec["training_reference"]:
            raise RuntimeError("Question/answer does not match original training split")
    return sp,audit,metrics

def as_examples(sp,audit,indices,max_len=192):
    result=[]
    for idx in indices:
        r=audit["results"][idx]
        pair=build_sft(sp,r["question"],r["training_reference"],max_len=max_len)
        if pair is None:raise RuntimeError("Training example exceeds context")
        result.append((r,pair))
    return result

@torch.no_grad()
def teacher_forced(model,pairs,pad_id):
    model.eval()
    results=[]
    for meta,pair in pairs:
        x,y=collate_lm([pair],pad_id)
        scores=model(x)
        ce=F.cross_entropy(scores.flatten(0,1),y.flatten(),
                           ignore_index=IGNORE,reduction="sum")
        tokens=int(y.ne(IGNORE).sum())
        results.append(float(ce)/max(1,tokens))
    return results

@torch.no_grad()
def generate_all(model,sp,pairs,seed=151):
    model.eval()
    out=[]
    for i,(meta,_) in enumerate(pairs):
        answer=model.generate(sp,meta["question"],
                              max_new_tokens=100,temperature=0,top_k=0,seed=seed+i)
        out.append({
          "question":meta["question"],"reference":meta["training_reference"],
          "answer":answer["text"],"eos":answer["eos"],
          "tokens":answer["tokens"],
          "exact":normalize(answer["text"])==normalize(meta["training_reference"]),
          "char_f1_proxy":round(char_f1(answer["text"],meta["training_reference"]),4)
        })
    return out

def summarize_outputs(records):
    return {"exact_count":sum(r["exact"] for r in records),
            "count":len(records),
            "avg_char_f1_proxy":round(sum(r["char_f1_proxy"] for r in records)/max(1,len(records)),4)}

def train(args):
    torch.set_num_threads(args.threads)
    rng=random.Random(args.seed)
    torch.manual_seed(args.seed)
    data=Path(args.data);base=Path(args.base);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    sp,audit,orig=verify_original(data,base,Path(args.audit))
    rehearsed=as_examples(sp,audit,REHEARSAL_INDICES)
    control=as_examples(sp,audit,CONTROL_INDICES)
    model,base_metadata=restore(base/"best.pt")
    before_train=teacher_forced(model,rehearsed,sp.pad_id())
    before_control=teacher_forced(model,control,sp.pad_id())
    before_outputs=generate_all(model,sp,rehearsed)
    before_controls=generate_all(model,sp,control)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0)
    count=[0]*len(rehearsed)
    order=list(range(len(rehearsed)))
    rng.shuffle(order)
    cursor=0
    traces=[]
    start=time.perf_counter()
    for step in range(1,args.steps+1):
        # Epoch cycling with no replacement: every example receives nearly
        # the same number of actual gradient-bearing exposures.
        selected=[]
        for _ in range(args.batch):
            if cursor>=len(order):
                rng.shuffle(order)
                cursor=0
            idx=order[cursor]
            cursor+=1
            selected.append(idx)
            count[idx]+=1
        x,y=collate_lm([rehearsed[idx][1] for idx in selected],sp.pad_id())
        model.train()
        logits=model(x)
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                             y.reshape(-1),ignore_index=IGNORE)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss at step "+str(step))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step==1 or step%args.report_every==0 or step==args.steps:
            now_train=teacher_forced(model,rehearsed,sp.pad_id())
            now_control=teacher_forced(model,control,sp.pad_id())
            trace={"step":step,"batch_loss":float(loss.detach()),
                   "mean_seen_nll":sum(now_train)/len(now_train),
                   "mean_control_nll":sum(now_control)/len(now_control),
                   "elapsed_seconds":round(time.perf_counter()-start,1)}
            traces.append(trace)
            print("FIT_PROGRESS",json.dumps(trace,ensure_ascii=False),flush=True)
    save(out/"trained.pt",model,{
        "base_checkpoint_sha256":hashlib.sha256((base/"best.pt").read_bytes()).hexdigest(),
        "original_training_step":base_metadata.get("global_step"),
        "target_training_steps":args.steps,
        "training_mode":"controlled seen-QA rehearsal; NOT open domain"
    })
    trained,metadata=restore(out/"trained.pt")
    after_train=teacher_forced(trained,rehearsed,sp.pad_id())
    after_control=teacher_forced(trained,control,sp.pad_id())
    after_outputs=generate_all(trained,sp,rehearsed)
    after_controls=generate_all(trained,sp,control)
    payload={
      "purpose":"Diagnose and correct inadequate sample exposure",
      "original_model":"AMNTMv3.1",
      "new_checkpoint":"fine-tuned original v3.1, preserved in independent file",
      "training_source_audit_verified":True,
      "training_source_file_sha256":orig["data"]["source_file_sha256"],
      "tokenizer_sha256":orig["tokenizer_sha256"],
      "original_steps":base_metadata.get("global_step"),
      "starting_checkpoint_sha256":metadata["base_checkpoint_sha256"],
      "fresh_optimization_steps":args.steps,
      "learning_rate":args.lr,"batch_size":args.batch,
      "trained_question_count":len(rehearsed),
      "original_answer_exposures":[audit["results"][i]["sft_exposures"] for i in REHEARSAL_INDICES],
      "new_answer_exposures":count,
      "heldout_from_new_rehearsal_question_count":len(control),
      "control_were_seen_in_original_v31":True,
      "not_evidence_of_open_domain_generalization":True,
      "before":{"train_teacher_forced_nll":before_train,
                "control_teacher_forced_nll":before_control,
                "seen_generation":before_outputs,
                "control_generation":before_controls,
                "seen_summary":summarize_outputs(before_outputs),
                "control_summary":summarize_outputs(before_controls)},
      "after":{"train_teacher_forced_nll":after_train,
               "control_teacher_forced_nll":after_control,
               "seen_generation":after_outputs,
               "control_generation":after_controls,
               "seen_summary":summarize_outputs(after_outputs),
               "control_summary":summarize_outputs(after_controls)},
      "history":traces,
      "train_seconds":round(time.perf_counter()-start,1)
    }
    (out/"metrics.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),
                                    encoding="utf-8")
    lines=[
      "# AMN v3.1 tekrarlı öğrenme — gerçek kaynak sorularıyla kontrol",
      "",
      "Modelin daha önce SFT aşamasında gördüğü ve gerçek Türkçe veri kümesinde bulunan",
      "altı soruya kontrollü ağırlık güncellemesi yapıldı. İki kontrol sorusu yeni",
      "öğretimde kullanılmadı. Bu test **ezberleme / optimizasyon** denetimidir;",
      "genel Türkçe anlama kanıtı değildir.","",
      "Orijinal checkpoint salt okunur kaldı. Kaynak SHA ve eğitim örneği kimliği",
      "doğrulandı. Çıktılar otomatik olarak üretildi, elle düzeltilmedi.",
      "",
      "Önce eğitim sorularında tam eşleşme: "+
       str(payload["before"]["seen_summary"]["exact_count"])+"/"+str(len(rehearsed)),
      "Sonra eğitim sorularında tam eşleşme: "+
       str(payload["after"]["seen_summary"]["exact_count"])+"/"+str(len(rehearsed)),
      "Ek eğitim adımı: "+str(args.steps),
      ""
    ]
    for i,item in enumerate(after_outputs):
        lines.extend(["## Öğretilen soru "+str(i+1),"**Soru:** "+item["question"],"",
                      "**Gerçek kaynak cevabı:** "+item["reference"],"",
                      "**AMN cevabı:** "+(item["answer"] or "(boş)"),"",
                      "Eğitimde tekrar gösterim: "+str(count[i]),
                      "Önceki NLL: "+f"{before_train[i]:.4f}"+
                      "; sonraki NLL: "+f"{after_train[i]:.4f}",""])
    for i,item in enumerate(after_controls):
        lines.extend(["## Ek eğitimde kullanılmayan kontrol "+str(i+1),
                      "**Soru:** "+item["question"],"",
                      "**Kaynak cevabı:** "+item["reference"],"",
                      "**AMN cevabı:** "+(item["answer"] or "(boş)"),"",
                      "Önceki NLL: "+f"{before_control[i]:.4f}"+
                      "; sonraki NLL: "+f"{after_control[i]:.4f}",""])
    (out/"raw_seen_qa_tr.md").write_text("\n".join(lines),encoding="utf-8")
    print("FINAL_RESULT",json.dumps({
      "before":payload["before"]["seen_summary"],
      "after":payload["after"]["seen_summary"],
      "control_before":payload["before"]["control_summary"],
      "control_after":payload["after"]["control_summary"],
      "sample_repeats":count,
      "model_saved":(out/"trained.pt").is_file()
    },ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/open_turkish_v31")
    p.add_argument("--base",default="results/amn_tm_v31")
    p.add_argument("--audit",default="results/amn_tm_v31/seen_training_questions/audit.json")
    p.add_argument("--output",default="results/amn_tm_v31_rehearsal")
    p.add_argument("--steps",type=int,default=360)
    p.add_argument("--batch",type=int,default=3)
    p.add_argument("--lr",type=float,default=0.0006)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--report-every",type=int,default=60)
    p.add_argument("--seed",type=int,default=73)
    train(p.parse_args())

if __name__=="__main__":main()
