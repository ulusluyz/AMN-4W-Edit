"""Run five-stage AMN-TM v3 training on real Turkish instruction data.

Stages: (1) Turkish answer LM (2) weak-intent labels (3) Q/A evidence
retrieval (4) synthetic two-turn prompts (5) SFT + auxiliary replay.
This is a BOUNDED CPU PILOT for the requested 16M/8k/512 target.
The effective training context, steps and token exposures are logged exactly;
it is not the planned 100-200M-token full-scale pretraining.

All old weights are kept. This model starts from scratch because its shape is
incompatible with the earlier 1.17M parameter AMN checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
import sentencepiece as spm

from amn_tm_v3 import (
    AMNTMv3,INTENTS,IGNORE,build_sft,build_answer_lm,
    collate_lm,collate_questions,save,restore,parameters
)

ROOT=Path(__file__).resolve().parents[1]
# These labels are HEURISTIC, not externally annotated Turkish intent truth.
# Report measured accuracy as rule-label agreement only, not human intent quality.
RULES=(
    ("calculation",("hesapla","kaçtır","kaç eder","yüzde","toplamı","formül","denklem")),
    ("comparison",("karşılaştır","farkları","arasındaki fark","hangisi daha","kıyasla")),
    ("writing",("hikaye yaz","şiir yaz","metin yaz","mektup yaz","taslak yaz")),
    ("procedure",("nasıl yapılır","adım adım","hangi adımları","kurulum","uygula","yöntemi")),
    ("explanation",("nedir","ne demek","neden","açıkla","anlat","niçin","ne işe")),
)

def intent_label(text):
    t=" ".join(text.casefold().split())
    for label,patterns in RULES:
        if any(p in t for p in patterns):
            return INTENTS.index(label)
    return INTENTS.index("other")

def read_split(folder,name):
    with (Path(folder)/(name+".jsonl")).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def train_tokenizer(path,training,vocab):
    folder=Path(path);folder.mkdir(parents=True,exist_ok=True)
    source=folder/"train_text_only.txt"
    with source.open("w",encoding="utf-8") as f:
        for x in training:
            f.write(("Kullanıcı: "+x["prompt"]+" Asistan: "+x["answer"])
                    .replace("\n"," ")+"\n")
    spm.SentencePieceTrainer.train(
        input=str(source),model_prefix=str(folder/"tokenizer"),
        model_type="bpe",vocab_size=vocab,pad_id=0,bos_id=1,eos_id=2,unk_id=3,
        byte_fallback=True,character_coverage=0.9995,hard_vocab_limit=False,
        num_threads=2,max_sentence_length=4096
    )
    source.unlink(missing_ok=True)
    sp=spm.SentencePieceProcessor(model_file=str(folder/"tokenizer.model"))
    return sp

def prepare_rows(sp,records,max_len):
    out=[]
    for record in records:
        sft=build_sft(sp,record["prompt"],record["answer"],max_len=max_len)
        lm=build_answer_lm(sp,record["answer"],max_len=max_len)
        if sft is not None and lm is not None:
            out.append({"prompt":record["prompt"],"answer":record["answer"],
                        "sft":sft,"lm":lm,"intent":intent_label(record["prompt"])})
    return out

@torch.no_grad()
def evaluate(model,rows,pad_id,batch=2,max_examples=32):
    model.eval()
    numerator=0.0;denominator=0;correct_intent=0;intent_total=0
    for i in range(0,min(max_examples,len(rows)),batch):
        subset=rows[i:i+batch]
        x,y=collate_lm([r["sft"] for r in subset],pad_id)
        logits=model(x)
        numerator+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                               y.reshape(-1),ignore_index=IGNORE,reduction="sum"))
        denominator+=int(y.ne(IGNORE).sum())
        q=collate_questions(sp_global,[r["prompt"] for r in subset],max_len=64)
        guessed=model.intent_logits(q).argmax(-1)
        targets=torch.tensor([r["intent"] for r in subset])
        correct_intent+=int((guessed==targets).sum())
        intent_total+=len(subset)
    nll=numerator/max(denominator,1)
    return {"answer_nll":nll,"answer_ppl":math.exp(min(30,nll)),
            "answer_tokens":denominator,
            "weak_intent_rule_agreement":correct_intent/max(1,intent_total),
            "weak_intent_n":intent_total}

def multi_turn_pair(sp,first,second,max_len):
    # Synthetic history of two UNRELATED real Turkish Q/A pairs.
    # This tests formatting, not true coherent multi-turn reasoning.
    question=(first["prompt"]+"\nAsistan: "+first["answer"]+
              "\nKullanıcı: "+second["prompt"])
    return build_sft(sp,question,second["answer"],max_len=max_len)

def choose_distinct(rng,rows,n):
    # Draw pairs from disjoint prompts and discourage identical answers.
    chosen=[];answers=set()
    for _ in range(n*20):
        rec=rows[rng.randrange(len(rows))]
        key=rec["answer"].strip().casefold()
        if key in answers:continue
        chosen.append(rec);answers.add(key)
        if len(chosen)>=n:break
    if len(chosen)!=n:raise ValueError("Cannot sample unique answer pairs")
    return chosen

class BalancedEpochSampler:
    """Visit each training example once per epoch before repeating it.

    Unlike random sampling with replacement, per-example exposure can be
    measured and controlled. For a 17,170-row dataset at batch 4, one SFT
    epoch requires ceil(17170/4) = 4,293 optimizer updates.
    """
    def __init__(self,rows,seed):
        self.rows=rows
        self.rng=random.Random(seed)
        self.order=list(range(len(rows)))
        self.rng.shuffle(self.order)
        self.pos=0
        self.exposures=[0]*len(rows)

    def take(self,n):
        if not 1<=n<=len(self.rows):
            raise ValueError("Invalid epoch batch size")
        selected=[]
        seen=set()
        while len(selected)<n:
            if self.pos==len(self.order):
                self.rng.shuffle(self.order)
                self.pos=0
            idx=self.order[self.pos]
            self.pos+=1
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(self.rows[idx])
            self.exposures[idx]+=1
        return selected

    def stats(self):
        samples=sum(self.exposures)
        covered=sum(x>0 for x in self.exposures)
        return {"total_samples":samples,"unique_samples":covered,
                "dataset_size":len(self.rows),
                "coverage_ratio":covered/len(self.rows),
                "minimum_exposures":min(self.exposures),
                "maximum_exposures":max(self.exposures)}


def train(args):
    global sp_global
    torch.set_num_threads(args.threads)
    random.seed(args.seed);torch.manual_seed(args.seed)
    src=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    original_manifest=json.loads((src/"manifest.json").read_text(encoding="utf-8"))
    train_raw=read_split(src,"train")
    dev_raw=read_split(src,"val")
    test_raw=read_split(src,"test")
    sp=train_tokenizer(out,train_raw,args.vocab)
    sp_global=sp
    train_rows=prepare_rows(sp,train_raw,args.max_len)
    dev_rows=prepare_rows(sp,dev_raw,args.max_len)
    test_rows=prepare_rows(sp,test_raw,args.max_len)
    if min(map(len,(train_rows,dev_rows,test_rows)))<20 or len(train_rows)<1000:
        raise RuntimeError("Insufficient real-text data after subword filtering")
    if args.model_version=="v3.1":
        from amn_tm_v31 import AMNTMv31,save as save_model,restore as restore_model
        model_class=AMNTMv31
    else:
        model_class=AMNTMv3
        save_model,restore_model=save,restore
    model=model_class(sp.vocab_size(),args.width,args.layers,args.key_dim,
                      args.semantic_dim,sp.pad_id())
    params=parameters(model)
    if args.width==544 and args.layers==6 and not 15_000_000<=params<=18_000_000:
        raise RuntimeError("Expected 16M-class architecture; got "+str(params))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.015)
    rng=random.Random(args.seed+200)
    sft_sampler=BalancedEpochSampler(train_rows,args.seed+12345) if args.sft_sampler=="epoch" else None
    sft_exposures=Counter()
    stages=[
        ("answer_language_model",args.lm_steps),
        ("weak_intent",args.intent_steps),
        ("in_batch_grounding",args.ground_steps),
        ("synthetic_multiturn",args.dialogue_steps),
        ("answer_sft_with_replay",args.sft_steps)
    ]
    tokens_lm=0;tokens_input=0;intent_examples=0;ground_pairs=0;multiturn_examples=0
    history=[];clock=time.perf_counter();completed=0;reason="all_stages"
    best_val=float("inf");best_step=0
    for stage,steps in stages:
        for i in range(steps):
            if time.perf_counter()-clock > args.max_seconds:
                reason="time_budget";break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if stage=="answer_language_model":
                sample=choose_distinct(rng,train_rows,args.batch)
                x,y=collate_lm([r["lm"] for r in sample],sp.pad_id())
                logits=model(x)
                loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                                     y.reshape(-1),ignore_index=IGNORE)
                tokens_lm+=int(y.ne(IGNORE).sum())
                tokens_input+=int(x.ne(sp.pad_id()).sum())
            elif stage=="weak_intent":
                sample=choose_distinct(rng,train_rows,args.batch)
                q=collate_questions(sp,[r["prompt"] for r in sample],max_len=args.semantic_len)
                targets=torch.tensor([r["intent"] for r in sample])
                loss=F.cross_entropy(model.intent_logits(q),targets)
                intent_examples+=len(sample)
                tokens_input+=int(q.ne(sp.pad_id()).sum())
            elif stage=="in_batch_grounding":
                sample=choose_distinct(rng,train_rows,max(2,args.retrieval_batch))
                q=collate_questions(sp,[r["prompt"] for r in sample],max_len=args.semantic_len)
                a=collate_questions(sp,[r["answer"] for r in sample],max_len=args.semantic_len)
                loss=model.retrieval_loss(q,a)
                ground_pairs+=len(sample)
                tokens_input+=int(q.ne(sp.pad_id()).sum()+a.ne(sp.pad_id()).sum())
            elif stage=="synthetic_multiturn":
                sample=[]
                for _ in range(args.batch):
                    for attempt in range(20):
                        first,second=choose_distinct(rng,train_rows,2)
                        item=multi_turn_pair(sp,first,second,args.max_len)
                        if item is not None:break
                    else:item=second["sft"]
                    sample.append(item)
                x,y=collate_lm(sample,sp.pad_id())
                logits=model(x)
                loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                                     y.reshape(-1),ignore_index=IGNORE)
                tokens_lm+=int(y.ne(IGNORE).sum())
                tokens_input+=int(x.ne(sp.pad_id()).sum())
                multiturn_examples+=len(sample)
            elif stage=="answer_sft_with_replay":
                sample=(sft_sampler.take(args.batch) if sft_sampler is not None
                        else choose_distinct(rng,train_rows,args.batch))
                for row in sample:
                    sft_exposures[row["prompt"]]+=1
                x,y=collate_lm([r["sft"] for r in sample],sp.pad_id())
                logits=model(x)
                loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),
                                     y.reshape(-1),ignore_index=IGNORE)
                tokens_lm+=int(y.ne(IGNORE).sum())
                tokens_input+=int(x.ne(sp.pad_id()).sum())
                if i%4==0:
                    q=collate_questions(sp,[r["prompt"] for r in sample],
                                         max_len=args.semantic_len)
                    a=collate_questions(sp,[r["answer"] for r in sample],
                                         max_len=args.semantic_len)
                    intent_targets=torch.tensor([r["intent"] for r in sample])
                    loss=loss+0.10*F.cross_entropy(model.intent_logits(q),intent_targets)
                    loss=loss+0.15*model.retrieval_loss(q,a)
                    intent_examples+=len(sample)
                    ground_pairs+=len(sample)
                    tokens_input+=int(q.ne(sp.pad_id()).sum()+a.ne(sp.pad_id()).sum())
            else:raise RuntimeError(stage)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss in "+stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            completed+=1
            if i==0 or (i+1)%args.log_every==0 or i==steps-1:
                row={"stage":stage,"stage_step":i+1,"global_step":completed,
                     "training_loss":float(loss.detach()),
                     "elapsed_seconds":round(time.perf_counter()-clock,2)}
                print("PROGRESS",json.dumps(row,ensure_ascii=False),flush=True)
                history.append(row)
            if stage=="answer_sft_with_replay" and ((i+1)%args.eval_every==0 or i==steps-1):
                metric=evaluate(model,dev_rows,sp.pad_id(),max_examples=args.eval_rows)
                print("DEVELOPMENT",completed,json.dumps(metric),flush=True)
                if metric["answer_nll"] < best_val:
                    best_val=metric["answer_nll"];best_step=completed
                    save_model(out/"best.pt",model,{"global_step":completed,
                         "stage":stage,"val_nll":best_val,"seed":args.seed})
        if reason=="time_budget":break
    # Prevent accidental quality claims from a zero-coverage training schedule.
    if args.require_sft_exposures>0:
        min_coverage=min(sft_sampler.exposures) if sft_sampler is not None else (
            min(sft_exposures.get(r["prompt"],0) for r in train_rows))
        if min_coverage<args.require_sft_exposures:
            raise RuntimeError(
                "INSUFFICIENT_SFT_COVERAGE: required each training question at least "+
                str(args.require_sft_exposures)+" times, minimum observed "+str(min_coverage)+
                "; increase --sft-steps or reduce data. Model not published.")

    # Ensure a checkpoint even when the time budget ends before the SFT stage.
    if not (out/"best.pt").exists():
        dev_metric=evaluate(model,dev_rows,sp.pad_id(),max_examples=args.eval_rows)
        best_val=dev_metric["answer_nll"];best_step=completed
        save_model(out/"best.pt",model,{"global_step":completed,"stage":"partial",
                                  "val_nll":best_val,"seed":args.seed})
    best_model,checkpoint_info=restore_model(out/"best.pt")
    test_result=evaluate(best_model,test_rows,sp.pad_id(),max_examples=args.test_rows)
    probes=(
        "Bir robot engellerden kaçınmak için hangi sensörleri kullanabilir?",
        "Python dilinde for döngüsü ne işe yarar?",
        "Elektrik devresinde Ohm kanunu nedir?",
        "Kablosuz ağda gecikme neden artar?",
        "Yeni bir yazılım projesinde riskler nasıl azaltılır?",
        "Türkçe öğrenmek için hangi yöntemler kullanılabilir?",
    )
    generated=[]
    for j,prompt in enumerate(probes):
        prediction=best_model.generate(sp,prompt,max_new_tokens=args.generate_tokens,
                                       temperature=.7,top_k=32,seed=args.seed+300+j)
        row={"question":prompt,**prediction}
        generated.append(row)
        print("GENERATED",json.dumps(row,ensure_ascii=False),flush=True)
    result={
        "architecture":"AMNTMv3.1" if args.model_version=="v3.1" else "AMNTMv3",
        "model_version":args.model_version,"config":best_model.config,
        "parameters":params,
        "experiment":"bounded real Turkish multi-objective CPU pilot",
        "targets_not_yet_achieved":{
            "full_pretraining_tokens":100_000_000,
            "natural_language_quality":"not established",
            "full_context_training":512,
            "human_intent_gold_labels":"not available",
            "true_multiturn_dataset":"not available"
        },
        "trained_from_scratch":True,
        "old_checkpoints_unchanged":True,
        "data":original_manifest,
        "data_counts":{"train":len(train_rows),"dev":len(dev_rows),"test":len(test_rows)},
        "tokenizer_vocab":sp.vocab_size(),
        "tokenizer_sha256":hashlib.sha256((out/"tokenizer.model").read_bytes()).hexdigest(),
        "effective_training_context_max":args.max_len,
        "configured_context_limit":512,
        "weak_intent_labels":True,
        "synthetic_multiturn_training":True,
        "scheduled_stages":dict(stages),
        "completed_updates":completed,"stop_reason":reason,
        "sft_sampling":args.sft_sampler,
        "sft_coverage":(sft_sampler.stats() if sft_sampler is not None else {
           "total_samples":sum(sft_exposures.values()),
           "unique_samples":len(sft_exposures),
           "dataset_size":len(train_rows),
           "coverage_ratio":len(sft_exposures)/len(train_rows),
           "minimum_exposures":min(sft_exposures.get(r["prompt"],0) for r in train_rows),
           "maximum_exposures":max(sft_exposures.values(),default=0)}),
        "train_lm_target_tokens":tokens_lm,
        "train_input_tokens":tokens_input,
        "weak_intent_supervision_examples":intent_examples,
        "contrastive_positive_pairs":ground_pairs,
        "synthetic_multiturn_examples":multiturn_examples,
        "best_checkpoint_step":best_step,
        "best_validation_answer_nll":best_val,
        "test_answer_metrics":test_result,
        "generated":generated,
        "history":history,
        "train_seconds":round(time.perf_counter()-clock,2)
    }
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),
                                    encoding="utf-8")
    lines=["# AMN-TM v3 — gerçek Türkçe üretim, beş görevli CPU pilotu","",
        "Cevaplar eğitilmiş model tarafından token-token üretilmiştir;",
        "hazır cevap veya ChatGPT düzeltmesi içermez.","",
        "Durum: "+reason+"; tamamlanan optimizasyon adımı: "+str(completed),
        "Parametre: "+str(params),
        "Gerçek LM hedef tokenı: "+str(tokens_lm),
        "Kullanılan azami eğitim bağlamı: "+str(args.max_len)+" token (hedef 512).",
        "Bağımsız test NLL: "+str(round(test_result["answer_nll"],4)),""]
    for i,row in enumerate(generated,1):
        lines.extend(["## Soru "+str(i),"**Soru:** "+row["question"],"",
                      "**Model cevabı:** "+(row["text"] or "(boş)"),"",
                      "EOS: "+str(row["eos"])+"; token: "+str(row["tokens"]),""])
    (out/"raw_turkish_outputs.md").write_text("\n".join(lines),encoding="utf-8")
    print("TRAINING_COMPLETE",json.dumps({"steps":completed,"params":params,
          "test_nll":test_result["answer_nll"],"reason":reason,
          "train_lm_target_tokens":tokens_lm},ensure_ascii=False),flush=True)
    return result

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--data",default="data/open_turkish_v3")
    a.add_argument("--model-version",choices=("v3","v3.1"),default="v3")
    a.add_argument("--output",default="results/amn_tm_v3_pilot")
    a.add_argument("--width",type=int,default=544)
    a.add_argument("--layers",type=int,default=6)
    a.add_argument("--key-dim",type=int,default=128)
    a.add_argument("--semantic-dim",type=int,default=192)
    a.add_argument("--vocab",type=int,default=8192)
    a.add_argument("--max-len",type=int,default=144)
    a.add_argument("--semantic-len",type=int,default=56)
    a.add_argument("--batch",type=int,default=2)
    a.add_argument("--retrieval-batch",type=int,default=4)
    a.add_argument("--sft-sampler",choices=("random","epoch"),default="random")
    a.add_argument("--require-sft-exposures",type=int,default=0)
    a.add_argument("--lr",type=float,default=0.0008)
    a.add_argument("--threads",type=int,default=2)
    a.add_argument("--seed",type=int,default=73)
    a.add_argument("--lm-steps",type=int,default=28)
    a.add_argument("--intent-steps",type=int,default=16)
    a.add_argument("--ground-steps",type=int,default=16)
    a.add_argument("--dialogue-steps",type=int,default=16)
    a.add_argument("--sft-steps",type=int,default=64)
    a.add_argument("--log-every",type=int,default=8)
    a.add_argument("--eval-every",type=int,default=16)
    a.add_argument("--eval-rows",type=int,default=24)
    a.add_argument("--test-rows",type=int,default=40)
    a.add_argument("--generate-tokens",type=int,default=112)
    a.add_argument("--max-seconds",type=int,default=2550)
    args=a.parse_args()
    train(args)

if __name__=="__main__":main()
