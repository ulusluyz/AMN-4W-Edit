"""Experience replay repair for AMN v3.1 after six-question memorization gate.

Mix the original six examples and both previously forgotten controls at every
balanced epoch. Evaluate exact answers for ALL eight before and after.
This demonstrates local retention, not unseen question generalization.
"""
from __future__ import annotations
import argparse,json,random,time,hashlib,sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_tm_v3 import collate_lm,IGNORE
from amn_tm_v31 import restore,save
from fix_amn_v31_rehearsal import (
    verify_original,as_examples,REHEARSAL_INDICES,CONTROL_INDICES,
    teacher_forced,generate_all,summarize_outputs
)

def mean(values):
    return sum(values)/max(1,len(values))

def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    base=Path(args.base)
    source=Path(args.data)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sp,audit,old=verify_original(source,Path(args.original),Path(args.audit))
    originals=as_examples(sp,audit,REHEARSAL_INDICES)
    controls=as_examples(sp,audit,CONTROL_INDICES)
    combined=originals+controls
    model,metadata=restore(base/"trained.pt")
    before_loss=teacher_forced(model,combined,sp.pad_id())
    before_generation=generate_all(model,sp,combined)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0)
    order=list(range(len(combined)));rng.shuffle(order);cursor=0
    exposure=[0]*len(combined)
    trace=[];started=time.perf_counter()
    for step in range(1,args.steps+1):
        picked=[]
        for _ in range(args.batch):
            if cursor>=len(order):
                rng.shuffle(order);cursor=0
            idx=order[cursor];cursor+=1
            picked.append(idx);exposure[idx]+=1
        x,y=collate_lm([combined[i][1] for i in picked],sp.pad_id())
        model.train()
        logits=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=IGNORE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step==1 or step%args.report_every==0 or step==args.steps:
            losses=teacher_forced(model,combined,sp.pad_id())
            row={"step":step,"training_loss":float(loss.detach()),
                 "six_old_seen_nll":mean(losses[:6]),
                 "two_replayed_control_nll":mean(losses[6:]),
                 "seconds":round(time.perf_counter()-started,1)}
            trace.append(row)
            print("REPLAY_STEP",json.dumps(row,ensure_ascii=False),flush=True)
    save(out/"trained.pt",model,{"parent_checkpoint_sha256":hashlib.sha256(
        (base/"trained.pt").read_bytes()).hexdigest(),
        "training_mode":"eight-source-question replay",
        "updates":args.steps})
    checkpoint,_=restore(out/"trained.pt")
    after_loss=teacher_forced(checkpoint,combined,sp.pad_id())
    after_generation=generate_all(checkpoint,sp,combined)
    result={
      "experiment":"replay all previously seen six + two controls",
      "scope":"exact learned/seen QAs, NOT open-domain generalization",
      "source_recreated_and_matched":True,
      "original_tokenizer_sha256":old["tokenizer_sha256"],
      "new_training_steps":args.steps,
      "batch":args.batch,"lr":args.lr,
      "per_question_replay_exposures":exposure,
      "old_six":{
         "before_nll":mean(before_loss[:6]),
         "after_nll":mean(after_loss[:6]),
         "before_summary":summarize_outputs(before_generation[:6]),
         "after_summary":summarize_outputs(after_generation[:6])
      },
      "previously_forgotten_two":{
         "before_nll":mean(before_loss[6:]),
         "after_nll":mean(after_loss[6:]),
         "before_summary":summarize_outputs(before_generation[6:]),
         "after_summary":summarize_outputs(after_generation[6:])
      },
      "before_answers":before_generation,
      "after_answers":after_generation,
      "history":trace,
      "training_seconds":round(time.perf_counter()-started,1)
    }
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# AMN v3.1 — sekiz gerçek soruyla unutmayı azaltma deneyi","",
         "Önceki altı öğrenilmiş soru ve eski iki kontrol sorusu birlikte",
         "dengeli tekrar eğitimine alındı. Orijinal ve altı-soruluk checkpointler",
         "değiştirilmedi. Çıktılar modelin gerçek otoregresif üretimidir.","",
         "Bu **görülmüş örnek / hafızada tutma** denetimidir; yeni sorulara",
         "genelleme veya genel Türkçe yeteneği kanıtlamaz.","",
         "Eski altı sorunun önce tam cevabı: "+
         str(result["old_six"]["before_summary"]["exact_count"])+"/6",
         "Eski altı sorunun sonra tam cevabı: "+
         str(result["old_six"]["after_summary"]["exact_count"])+"/6",
         "Önceki iki kontrolün önce tam cevabı: "+
         str(result["previously_forgotten_two"]["before_summary"]["exact_count"])+"/2",
         "Önceki iki kontrolün sonra tam cevabı: "+
         str(result["previously_forgotten_two"]["after_summary"]["exact_count"])+"/2",""]
    for i,rec in enumerate(after_generation):
        lines.extend(["## Soru "+str(i+1), "**Girdi:** "+rec["question"],"",
                      "**Eğitim verisindeki hedef:** "+rec["reference"],"",
                      "**AMN'nin cevabı:** "+(rec["answer"] or "(boş)"),"",
                      "Ek tekrar: "+str(exposure[i])+
                      "; önce NLL: "+f"{before_loss[i]:.4f}"+
                      "; sonra NLL: "+f"{after_loss[i]:.4f}",""])
    (out/"replay_results_tr.md").write_text("\n".join(lines),encoding="utf-8")
    print("FINAL_REPLAY",json.dumps({
        "six_before":result["old_six"]["before_summary"],
        "six_after":result["old_six"]["after_summary"],
        "two_before":result["previously_forgotten_two"]["before_summary"],
        "two_after":result["previously_forgotten_two"]["after_summary"],
        "exposures":exposure
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",default="results/amn_tm_v31")
    p.add_argument("--base",default="results/amn_tm_v31_rehearsal")
    p.add_argument("--audit",default="results/amn_tm_v31/seen_training_questions/audit.json")
    p.add_argument("--data",default="data/open_turkish_v31")
    p.add_argument("--output",default="results/amn_tm_v31_replay8")
    p.add_argument("--steps",type=int,default=320)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--lr",type=float,default=0.0004)
    p.add_argument("--report-every",type=int,default=80)
    p.add_argument("--seed",type=int,default=74)
    p.add_argument("--threads",type=int,default=2)
    run(p.parse_args())
