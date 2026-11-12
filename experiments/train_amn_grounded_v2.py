"""Continue existing AMN v3.1 world-knowledge weights for six balanced epochs.

NO downloads and NO rebuild of previous training data. Preserve every existing
checkpoint. Train broad SFT 6 passes (early stop on truly disjoint validation),
then train EXISTING AMN query/evidence heads with topic-distinct contrastive
pairs. Deploy with source-attributed local memory as an evidence gate.

Important: the offline fact answer is extractively retrieved, not passed off
as neural-generation. A memorization-only gain is not generalization.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_tm_v31 import restore,save
from amn_tm_v3 import build_sft,collate_lm,collate_questions,IGNORE
from train_amn_tm_v3 import BalancedEpochSampler
from amn_grounded_memory_v1 import MemoryIndex,NOT_FOUND

QUESTIONS=[
    "Kedi nedir?","Köpek nedir?","Denklem nedir?",
    "İnsan nedir?","Erkek nedir?","Kadın nedir?",
    "20 tane erkek insan 8 tane kadın insan ismi yaz",
    "Cumhuriyet nedir","Güneş nedir","Salata nasıl yapılır",
    "Saksı nedir","Pusula nedir",
    "Kedi nasıl kullanılır",
]

def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()
            if s.strip()]

def verify_data(data):
    data=Path(data)
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    audit=json.loads((data/"audit.json").read_text(encoding="utf-8"))
    sha=hash_file(data/"downloaded_raw_pages.jsonl")
    if sha!=manifest["source_file_sha256"] or sha!=audit["source_sha256"]:
        raise RuntimeError("Existing cleaned Wikipedia source checksum changed")
    if audit["accepted_page_count"]<200 or audit["rejected_page_count"]<50:
        raise RuntimeError("Unverified data audit")
    splits={k:read_rows(data/(k+".jsonl")) for k in ("train","val","test")}
    topic_split={}
    for split,rows in splits.items():
        if len(rows)!=manifest[split]:raise RuntimeError("Manifest split count mismatch")
        for r in rows:
            if not r.get("source_url") or not r.get("answer"):
                raise RuntimeError("Missing provenance or answer")
            if r["source"]=="Turkish Wikipedia":
                previous=topic_split.setdefault(r["topic"],split)
                if previous!=split:raise RuntimeError("Data leakage across topic holdouts")
            elif split!="train":
                raise RuntimeError("Non-Wikipedia factual QA outside train partition")
    return splits,manifest,audit

def prepare(sp,records,max_length):
    result=[]
    for r in records:
        pair=build_sft(sp,r["prompt"],r["answer"],max_len=max_length)
        if pair:
            result.append({**r,"sft":pair})
    if len(result)<len(records)*.93:
        raise RuntimeError("Too many existing QA examples excluded after tokenization")
    return result

@torch.inference_mode()
def compute_nll(model,data,pad,batch=4):
    model.eval()
    total=0.;n=0
    for start in range(0,len(data),batch):
        inp,target=collate_lm([r["sft"] for r in data[start:start+batch]],pad)
        logits=model(inp)
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                             target.reshape(-1),ignore_index=IGNORE,reduction="sum")
        total+=float(loss)
        n+=int(target.ne(IGNORE).sum())
    return total/max(1,n)

def grouped_wiki(records):
    groups=defaultdict(list)
    for r in records:
        if r["source"]=="Turkish Wikipedia":
            groups[r["topic"]].append(r)
    return groups

def contrastive_loss_from_frozen_backbone(model,sp,items):
    q=collate_questions(sp,[r["prompt"] for r in items],max_len=72)
    a=collate_questions(sp,[r["answer"] for r in items],max_len=92)
    with torch.no_grad():
        qh=model.pooled(q)
        ah=model.pooled(a)
    qz=F.normalize(model.query_head(qh),dim=-1)
    az=F.normalize(model.evidence_head(ah),dim=-1)
    scores=(qz@az.T)/.15
    targets=torch.arange(len(items))
    loss=(F.cross_entropy(scores,targets)+F.cross_entropy(scores.T,targets))/2
    return loss,scores

@torch.inference_mode()
def retrieval_recall(model,sp,groups,sample_pairs=96,batch=8):
    # Compute in-batch recall@1 over distinct, never-trained topics.
    # Do not conflate in-batch recall with a search over all documents.
    topics=sorted(groups)
    if len(topics)<8:
        return {"recall_at1_inbatch":None,"questions":0}
    rng=random.Random(811)
    if len(topics)>sample_pairs:topics=rng.sample(topics,sample_pairs)
    correct=0;tested=0
    for start in range(0,len(topics),batch):
        pack=topics[start:start+batch]
        if len(pack)<2:continue
        recs=[groups[x][0] for x in pack]
        q=collate_questions(sp,[r["prompt"] for r in recs],max_len=72)
        a=collate_questions(sp,[r["answer"] for r in recs],max_len=92)
        qs=F.normalize(model.query_head(model.pooled(q)),dim=-1)
        ans=F.normalize(model.evidence_head(model.pooled(a)),dim=-1)
        scores=qs@ans.T
        correct+=int((scores.argmax(-1)==torch.arange(len(recs))).sum())
        tested+=len(recs)
    return {"recall_at1_inbatch":correct/max(1,tested),"questions":tested}

def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    begin=time.perf_counter()
    source=Path(args.data);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    splits,manifest,audit=verify_data(source)
    sp=spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    data={k:prepare(sp,v,args.max_length) for k,v in splits.items()}
    assert len(data["train"])>600 and len(data["val"])>20
    old_sha=hash_file(args.base)
    old_model,_=restore(args.base)
    if sp.vocab_size()!=old_model.config["vocab_size"]:raise RuntimeError("Wrong tokenizer")
    train=data["train"];val=data["val"];test=data["test"]
    wiki_train=grouped_wiki(train)
    wiki_val=grouped_wiki(val)
    wiki_test=grouped_wiki(test)
    initial_val=compute_nll(old_model,val,sp.pad_id())
    initial_test=compute_nll(old_model,test,sp.pad_id())
    initial_retrieval=retrieval_recall(old_model,sp,wiki_val)
    print("START",json.dumps({"val_nll":initial_val,"test_nll":initial_test,
              "train_rows":len(train),"wiki_topics":len(wiki_train),
              "val_retrieval":initial_retrieval},ensure_ascii=False),flush=True)
    model=old_model
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.003)
    base_sampler=BalancedEpochSampler(train,args.seed+1001)
    core=[r for r in train if r["category"] in ("resmi_isimler","uzman_kaynakli_temel_kavram")
          or r["topic"] in ("Köpek","Kedi","İnsan","Denklem","Güneş","Erkek","Kadın")]
    core_sampler=BalancedEpochSampler(core,args.seed+1002)
    updates_per_epoch=math.ceil(len(train)/args.batch)
    best_nll=initial_val;best_step=0;current_step=0;bad_epochs=0;history=[]
    stop_reason="planned_six_epochs";completed_epochs=0
    # Snapshot baseline if no new epoch improves heldout validation.
    save(output/"best_sft.pt",model,{"stage":"baseline","parent_sha256":old_sha,
                                   "step":0,"val_nll":initial_val})
    for epoch in range(1,args.epochs+1):
        for j in range(updates_per_epoch):
            if time.perf_counter()-begin>args.max_seconds:
                stop_reason="cpu_time_limit";break
            broad=base_sampler.take(args.batch)
            extras=core_sampler.take(args.core_batch)
            x,y=collate_lm([r["sft"] for r in broad+extras],sp.pad_id())
            model.train()
            logits=model(x)
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                                 y.reshape(-1),ignore_index=IGNORE)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite model loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            current_step+=1
        if stop_reason=="cpu_time_limit":
            break
        completed_epochs+=1
        nll=compute_nll(model,val,sp.pad_id())
        rec={"epoch":epoch,"updates":current_step,
             "validation_nll":nll,
             "broad_coverage":base_sampler.stats(),
             "core_replay_coverage":core_sampler.stats(),
             "elapsed_s":round(time.perf_counter()-begin,1)}
        history.append(rec)
        print("EPOCH",json.dumps(rec,ensure_ascii=False),flush=True)
        if nll<best_nll-args.min_delta:
            best_nll=nll;best_step=current_step;bad_epochs=0
            save(output/"best_sft.pt",model,{"stage":"balanced_sft",
              "epoch":epoch,"step":current_step,"parent_sha256":old_sha,"val_nll":nll})
        else:
            bad_epochs+=1
            if bad_epochs>=args.patience:
                stop_reason="early_stopping_no_val_improvement";break
    # Full-pass gate: never advertise data-wide learning unless completed.
    actual=base_sampler.stats()
    if completed_epochs==0:
        raise RuntimeError("Training was interrupted before first complete data epoch")
    checkpoint,base_meta=restore(output/"best_sft.pt")
    best_before_retrieval=retrieval_recall(checkpoint,sp,wiki_val)
    # Preserve autoregressive weights; learn only EXISTING neural retrieval
    # heads for different-topic positive/negative comparisons.
    for param in checkpoint.parameters():
        param.requires_grad_(False)
    for head in (checkpoint.query_head,checkpoint.evidence_head):
        for param in head.parameters():param.requires_grad_(True)
    retrieval_optimizer=torch.optim.AdamW(
        list(checkpoint.query_head.parameters())+list(checkpoint.evidence_head.parameters()),
        lr=args.retrieval_lr,weight_decay=.01)
    topic_names=sorted(wiki_train)
    retrieval_history=[];best_recall=best_before_retrieval["recall_at1_inbatch"]
    best_retrieval_step=0
    # Store initial neural heads before contrastive training.
    best_q={k:v.detach().clone() for k,v in checkpoint.query_head.state_dict().items()}
    best_a={k:v.detach().clone() for k,v in checkpoint.evidence_head.state_dict().items()}
    for step in range(1,args.retrieval_steps+1):
        if time.perf_counter()-begin>args.max_seconds+150:
            break
        topics=rng.sample(topic_names,args.retrieval_batch)
        chosen=[rng.choice(wiki_train[t]) for t in topics]
        checkpoint.train()
        loss,scores=contrastive_loss_from_frozen_backbone(checkpoint,sp,chosen)
        retrieval_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(checkpoint.query_head.parameters())+
            list(checkpoint.evidence_head.parameters()),1.0)
        retrieval_optimizer.step()
        if step==1 or step%args.retrieval_eval_every==0 or step==args.retrieval_steps:
            checkpoint.eval()
            recall=retrieval_recall(checkpoint,sp,wiki_val)
            score=recall["recall_at1_inbatch"]
            row={"step":step,"train_contrastive_nll":float(loss.detach()),
                 "val_inbatch_recall_at1":score}
            retrieval_history.append(row)
            print("RETRIEVAL",json.dumps(row,ensure_ascii=False),flush=True)
            if score is not None and score>best_recall:
                best_recall=score;best_retrieval_step=step
                best_q={k:v.detach().clone() for k,v in checkpoint.query_head.state_dict().items()}
                best_a={k:v.detach().clone() for k,v in checkpoint.evidence_head.state_dict().items()}
    checkpoint.query_head.load_state_dict(best_q)
    checkpoint.evidence_head.load_state_dict(best_a)
    for p in checkpoint.parameters():p.requires_grad_(True)
    save(output/"best.pt",checkpoint,{"stage":"memory_reranker",
      "parent_sha256":old_sha,"sft_step":best_step,
      "retrieval_step":best_retrieval_step,
      "val_inbatch_recall":best_recall,
      "train_data_sha256":hash_file(source/"train.jsonl")})
    complete,meta=restore(output/"best.pt")
    val_end=compute_nll(complete,val,sp.pad_id())
    test_end=compute_nll(complete,test,sp.pad_id())
    retrieval_test=retrieval_recall(complete,sp,wiki_test)
    memory=MemoryIndex(train_path=source/"train.jsonl",model=complete,tokenizer=sp)
    results=[memory.answer(q).as_dict() for q in QUESTIONS]
    # Show the user real examples from downloaded, already-reviewed train data,
    # not just their own fixed probes.
    training_examples=[]
    for topic in ("Köpek","Kedi","İnsan","Denklem","Erkek","Kadın"):
        item=next(r for r in train if r["topic"]==topic)
        training_examples.append({"question":item["prompt"],"answer":item["answer"],
                                  "source_url":item["source_url"]})
    report={
      "objective":"offline question-to-verified-evidence memory + six-epoch SFT and neural contrastive ranker",
      "did_download_data":False,
      "changed_original_checkpoint":False,
      "was_neural_generation":False,
      "backbone":"AMNTMv3.1; 16380518 parameters; Transformer-free",
      "original_checkpoint_sha256":old_sha,
      "training_source_sha256":hash_file(source/"train.jsonl"),
      "train_rows":len(train),"validation_rows":len(val),"test_rows":len(test),
      "source_qa_count":len(splits["train"]),
      "planned_epochs":args.epochs,"epochs_completed":completed_epochs,
      "sft_training_updates":current_step,
      "best_sft_step":best_step,
      "stop_reason":stop_reason,
      "broad_exposure":actual,
      "core_exposure":core_sampler.stats(),
      "initial_validation_nll":initial_val,"best_validation_nll":best_nll,
      "initial_test_nll":initial_test,
      "final_test_nll":test_end,
      "inbatch_retrieval_val_before":initial_retrieval,
      "inbatch_retrieval_val_after_sft":best_before_retrieval,
      "inbatch_retrieval_val_after":best_recall,
      "inbatch_retrieval_test_after":retrieval_test,
      "retrieval_steps_requested":args.retrieval_steps,
      "retrieval_best_step":best_retrieval_step,
      "history":history,
      "retrieval_history":retrieval_history,
      "source_data_examples":training_examples,
      "user_question_results":results,
      "source_limit":"Only train split included in factual memory. Unmatched questions abstain. Answers are retrieved source text, not generated.",
      "train_seconds":round(time.perf_counter()-begin,1)
    }
    (output/"metrics.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# AMN v3.1 — Bilgi belleği, soru eşleştirme ve cevap koruması","",
           "Kaynak yalnızca mevcut temiz eğitim verisidir; veri indirimi yapılmadı.",
           "Doğrulanmış bir kayıt bulunursa cevap doğrudan BELLEKTEKİ KAYNAK metnidir.",
           "Bu cevaplar sinir ağının serbest metin üretimi olarak sunulmamıştır.",
           "Eşleşme yoksa alakasız kelimeler üretmek yerine bilgi yok denir.","",
           "Eğitim turu: "+str(completed_epochs)+"/"+str(args.epochs),
           "Yeni SFT adımı: "+str(current_step),
           "SFT doğrulama kaybı: "+f"{initial_val:.4f} -> {best_nll:.4f}",
           "Bağımsız test kaybı: "+f"{initial_test:.4f} -> {test_end:.4f}",
           "Öğrenilmiş kanıt eşleştirme val in-batch recall@1: "+
                   str(initial_retrieval["recall_at1_inbatch"])+" -> "+str(best_recall),
           "","## Veri kümesinde gerçekten ne var?",""]
    for r in training_examples:
        lines.extend(["**"+r["question"]+"**",r["answer"],r["source_url"],""])
    lines+=["","## Kullanıcı soruları ve belleğin yanıtları",""]
    for r in results:
        lines.extend(["**"+r["question"]+"**",r["answer"],
                      "Eşleştirme: "+r["method"]+"; konu: "+(r["topic"] or "bilinmiyor")+
                      "; kaynak: "+(r["source_url"] or "yok"),""])
    (output/"memory_questions_tr.md").write_text("\n".join(lines),encoding="utf-8")
    print("COMPLETE",json.dumps({"epoch":completed_epochs,"step":current_step,
          "best_sft_step":best_step,"val_nll":best_nll,
          "test_before":initial_test,"test_after":test_end,
          "retrieval_recall":best_recall,"probe":results},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/amn_world_v1")
    p.add_argument("--base",default="results/amn_world_v1_train/best.pt")
    p.add_argument("--tokenizer",default="results/amn_tm_v31/tokenizer.model")
    p.add_argument("--output",default="results/amn_world_v2_grounded")
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--core-batch",type=int,default=2)
    p.add_argument("--max-length",type=int,default=192)
    p.add_argument("--max-seconds",type=int,default=2300)
    p.add_argument("--min-delta",type=float,default=.015)
    p.add_argument("--patience",type=int,default=2)
    p.add_argument("--lr",type=float,default=.00013)
    p.add_argument("--retrieval-lr",type=float,default=.0015)
    p.add_argument("--retrieval-batch",type=int,default=10)
    p.add_argument("--retrieval-steps",type=int,default=240)
    p.add_argument("--retrieval-eval-every",type=int,default=40)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=177)
    run(p.parse_args())

if __name__=="__main__":
    main()
