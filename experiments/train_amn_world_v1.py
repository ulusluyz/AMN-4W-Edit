"""General-knowledge SFT for native 16.38M AMN, starting from original v3.1.

Trains ONLY on previously quality-audited Turkish Wikipedia facts and official
2025 TÜİK baby names. Topic-disjoint heldout evaluation is preserved.
Balanced full-epoch SFT plus additional basic-fact replay (explicitly counted).
No Transformer, no answer lookup at inference, no false generalization claims.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,time,sys
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_tm_v31 import restore,save
from amn_tm_v3 import build_sft,collate_lm,IGNORE
from train_amn_tm_v3 import BalancedEpochSampler

DIRECT_QUESTIONS=[
    "Web geliştirme nedir","Teknik yön nedir","Denklem nedir",
    "Köpek nedir","10.000 kuş türü mü vardır",
    "Kadın kime denir","Erkek kime denir","İnsan cinsiyetleri neler",
    "20 tane erkek insan 8 tane kadın insan ismi yaz",
]
CORE={"Köpek","Kedi","İnsan","Kadın","Erkek","Denklem","Web geliştirme",
      "Web tasarımı","Bilgisayar","Atom","Güneş","Dünya","İnternet","Yapay zekâ",
      "Robot","Elektrik","Cinsiyet","Toplumsal cinsiyet","Matematik","Beyin"}

def load_jsonl(path):
    with open(path,encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_data(folder):
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    audit=json.loads((folder/"audit.json").read_text(encoding="utf-8"))
    raw=folder/"downloaded_raw_pages.jsonl"
    if fingerprint(raw)!=manifest["source_file_sha256"] or fingerprint(raw)!=audit["source_sha256"]:
        raise RuntimeError("Wikipedia raw source SHA mismatch")
    if not audit.get("topic_disjoint") or audit["accepted_page_count"]<90:
        raise RuntimeError("Quality gate metadata absent")
    rows={name:load_jsonl(folder/(name+".jsonl")) for name in ("train","val","test")}
    topics={}
    for split,records in rows.items():
        if len(records)!=manifest[split]:raise RuntimeError("Split counts mismatch")
        for r in records:
            if not r["prompt"] or not r["answer"]:raise RuntimeError("Empty QA")
            if r["source"]=="Turkish Wikipedia":
                if not r["source_url"].startswith("https://tr.wikipedia.org/wiki/"):
                    raise RuntimeError("Unattributed Wikipedia source")
                if r["topic"] in topics and topics[r["topic"]]!=split:
                    raise RuntimeError("Topic leakage "+r["topic"])
                topics[r["topic"]]=split
            elif r["source"]=="TÜİK ADNKS 2025":
                if split!="train":raise RuntimeError("Official names partition leak")
            else:raise RuntimeError("Unapproved source "+r["source"])
    return manifest,audit,rows

def tokenize(sp,records,max_length):
    encoded=[];skipped=[]
    for item in records:
        pair=build_sft(sp,item["prompt"],item["answer"],max_len=max_length)
        if pair is None:
            skipped.append(item["topic"])
            continue
        encoded.append({**item,"sft":pair})
    return encoded,skipped

@torch.no_grad()
def test_nll(model,rows,pad_id,batch_size=4,limit=80):
    model.eval()
    total=0.0;tokens=0
    for i in range(0,min(len(rows),limit),batch_size):
        subset=rows[i:i+batch_size]
        x,y=collate_lm([v["sft"] for v in subset],pad_id)
        logits=model(x)
        total+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),
                                      ignore_index=IGNORE,reduction="sum"))
        tokens+=int((y!=IGNORE).sum())
    nll=total/max(1,tokens)
    return {"nll":nll,"ppl":math.exp(min(30,nll)),"tokens":tokens,
            "sample_count":min(limit,len(rows))}

@torch.no_grad()
def probe(model,sp,questions,max_new=100):
    model.eval()
    outputs=[]
    for i,q in enumerate(questions):
        gen=model.generate(sp,q,max_new_tokens=max_new,
                           temperature=0,top_k=0,seed=901+i)
        outputs.append({"question":q,"answer":gen["text"],
                        "generated_tokens":gen["tokens"],"eos":gen["eos"]})
    return outputs

def core_sample(r):
    return r["category"]=="resmi_isimler" or r["topic"] in CORE

def train(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed);torch.manual_seed(args.seed)
    folder=Path(args.data);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    manifest,audit,splits=validate_data(folder)
    if not Path(args.base).is_file() or not Path(args.tokenizer).is_file():
        raise FileNotFoundError("Original pretrained AMN v3.1 weights/tokenizer not found")
    sp=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    encoded={};dropped={}
    for split,rows in splits.items():
        encoded[split],dropped[split]=tokenize(sp,rows,args.max_length)
        if len(dropped[split])>len(rows)*0.07:
            raise RuntimeError("Too much source data lost during real tokenizer encoding")
    tr=encoded["train"];dev=encoded["val"];te=encoded["test"]
    if len(tr)<160 or min(len(dev),len(te))<20:
        raise RuntimeError("Insufficient quality-gated training/heldout examples")
    cores=[r for r in tr if core_sample(r)]
    if len(cores)<12:raise RuntimeError("Missing basic fact / official names")
    model,base_info=restore(args.base)
    if sp.vocab_size()!=model.config["vocab_size"]:
        raise RuntimeError("Tokenizer and pretrained checkpoint mismatch")
    base_sha=fingerprint(args.base)
    tokenizer_sha=fingerprint(args.tokenizer)
    train_by_prompt={r["prompt"] for r in tr}
    before={"val":test_nll(model,dev,sp.pad_id(),limit=args.eval_rows),
            "test":test_nll(model,te,sp.pad_id(),limit=args.test_rows),
            "direct":probe(model,sp,DIRECT_QUESTIONS)}
    print("BASELINE",json.dumps({"val_nll":before["val"]["nll"],
           "test_nll":before["test"]["nll"]},ensure_ascii=False),flush=True)

    broad_sampler=BalancedEpochSampler(tr,args.seed+900)
    core_sampler=BalancedEpochSampler(cores,args.seed+901)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.003)
    clock=time.perf_counter();step=0;stop_reason="epochs_completed"
    logs=[];best_val=float("inf");best_step=0;completed_epochs=0
    updates_per_epoch=math.ceil(len(tr)/args.general_batch)
    max_updates=min(args.max_steps,args.epochs*updates_per_epoch)
    if max_updates<updates_per_epoch:raise RuntimeError("Cannot fit one full data epoch into max steps")
    for i in range(max_updates):
        if time.perf_counter()-clock>=args.max_seconds:
            stop_reason="time_budget";break
        main_batch=broad_sampler.take(args.general_batch)
        core_batch=core_sampler.take(args.core_batch)
        samples=main_batch+core_batch
        x,y=collate_lm([r["sft"] for r in samples],sp.pad_id())
        model.train()
        logits=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=IGNORE)
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss at step "+str(i+1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        step+=1
        if step%updates_per_epoch==0:completed_epochs+=1
        if step==1 or step%args.log_every==0 or step==max_updates:
            log={"step":step,"training_nll":float(loss.detach()),
                 "elapsed_seconds":round(time.perf_counter()-clock,1),
                 "broad_coverage":round(broad_sampler.stats()["coverage_ratio"],4)}
            logs.append(log)
            print("TRAIN",json.dumps(log,ensure_ascii=False),flush=True)
        if step%args.eval_every==0 or step==max_updates:
            dev_metric=test_nll(model,dev,sp.pad_id(),limit=args.eval_rows)
            print("VALIDATE",step,dev_metric["nll"],flush=True)
            if dev_metric["nll"]<best_val:
                best_val=dev_metric["nll"];best_step=step
                save(out/"best.pt",model,{"source":"quality-gated Turkish Wikipedia/TUIK",
                      "step":step,"validation_nll":best_val,
                      "parent_checkpoint_sha256":base_sha,
                      "tokenizer_sha256":tokenizer_sha})
    if best_step==0:
        best_val=test_nll(model,dev,sp.pad_id(),limit=args.eval_rows)["nll"]
        best_step=step
        save(out/"best.pt",model,{"step":step,"validation_nll":best_val,
             "parent_checkpoint_sha256":base_sha,"tokenizer_sha256":tokenizer_sha})
    finished,meta=restore(out/"best.pt")
    after={"val":test_nll(finished,dev,sp.pad_id(),limit=args.eval_rows),
           "test":test_nll(finished,te,sp.pad_id(),limit=args.test_rows),
           "direct":probe(finished,sp,DIRECT_QUESTIONS)}
    known=sorted({r["prompt"] for r in tr if r["topic"] in CORE})
    focus=("Köpek nedir?","Denklem nedir?","Kadın nedir?","Erkek nedir?",
           "İnsan nedir?","Bilgisayar nedir?","Robot nedir?","Dünya nedir?")
    after["canonical"]=probe(finished,sp,focus)
    report={
      "training":"genuine CPU full-epoch supervised AMN v3.1 knowledge adaptation",
      "trained":step>0,"architecture":"AMNTMv3.1",
      "source_manifest":manifest,"quality_audit":{
          "accepted_pages":audit["accepted_page_count"],
          "rejected_pages":audit["rejected_page_count"],
          "rejection_reasons":audit["rejection_reasons"],
          "category_counts":audit["accepted_by_category"]},
      "pretrained_checkpoint":str(args.base),"pretrained_sha256":base_sha,
      "tokenizer_sha256":tokenizer_sha,
      "input_rows":{k:len(v) for k,v in splits.items()},
      "encoded_rows":{k:len(v) for k,v in encoded.items()},
      "dropped_by_max_length":{k:len(v) for k,v in dropped.items()},
      "max_training_tokens":args.max_length,
      "model_parameters":sum(p.numel() for p in finished.parameters()),
      "seed":args.seed,"learning_rate":args.lr,
      "main_batch":args.general_batch,"core_batch":args.core_batch,
      "core_topic_count":len(cores),"total_updates":step,"completed_epochs":completed_epochs,
      "stop_reason":stop_reason,"best_step":best_step,
      "best_validation_nll":best_val,
      "main_source_coverage":broad_sampler.stats(),
      "core_replay_coverage":core_sampler.stats(),
      "full_dataset_one_pass_completed":broad_sampler.stats()["minimum_exposures"]>=1,
      "before":before,"after":after,"history":logs,
      "quality_limit":"Wikipedia article facts have programmatic screening, not per-fact human verification; exact QA success does not prove general intelligence",
      "train_seconds":round(time.perf_counter()-clock,1)
    }
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    direct_lines=[
       "# AMN v3.1 — Temiz Türkçe dünya bilgisi eğitimi sonrası gerçek cevaplar","",
       "Bu yanıtlar modelin checkpointinden greedy decoding ile üretilmiştir.",
       "Yapay örnek cevabı olarak değiştirilmemiştir. Tam bir genel bilgi modeli değildir.","",
       "Model güncellemesi: "+str(step)+"; en iyi ağırlık adımı: "+str(best_step),
       "Bağımsız test NLL: "+f"{before['test']['nll']:.4f}"+
       " -> "+f"{after['test']['nll']:.4f}",""]
    for row in after["direct"]:
        pre=next(r for r in before["direct"] if r["question"]==row["question"])
        direct_lines += ["## "+row["question"],"**Önce:** "+(pre["answer"] or "(boş)"),"",
                         "**Sonra:** "+(row["answer"] or "(boş)"),""]
    direct_lines += ["","## Doğrudan eğitimle desteklenen tanımlar"]
    for row in after["canonical"]:
        direct_lines += ["**"+row["question"]+"**",(row["answer"] or "(boş)"),""]
    (out/"raw_direct_answers_tr.md").write_text("\n".join(direct_lines),encoding="utf-8")
    print("TRAIN_END",json.dumps({"steps":step,"full_epochs":completed_epochs,
        "source_coverage":report["main_source_coverage"],
        "test_nll_before":before["test"]["nll"],"test_nll_after":after["test"]["nll"],
        "stop":stop_reason},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/amn_world_v1")
    p.add_argument("--base",default="results/amn_tm_v31/best.pt")
    p.add_argument("--tokenizer",default="results/amn_tm_v31/tokenizer.model")
    p.add_argument("--output",default="results/amn_world_v1_train")
    p.add_argument("--max-length",type=int,default=192)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--max-steps",type=int,default=1400)
    p.add_argument("--max-seconds",type=int,default=2500)
    p.add_argument("--lr",type=float,default=0.00025)
    p.add_argument("--general-batch",type=int,default=2)
    p.add_argument("--core-batch",type=int,default=2)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--eval-rows",type=int,default=60)
    p.add_argument("--test-rows",type=int,default=80)
    p.add_argument("--eval-every",type=int,default=160)
    p.add_argument("--log-every",type=int,default=120)
    p.add_argument("--seed",type=int,default=131)
    train(p.parse_args())
