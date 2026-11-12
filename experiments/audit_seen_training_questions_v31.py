"""Inspect exact AMN v3.1 training data, replay deterministic SGD draws, ask seen questions.

Only answer-SFT examples used for actual gradient updates are chosen.
Provenance verified by comparing source/tokenizer hashes and training token
counters against the archived v3.1 metrics. No training or weight mutation.
"""
from __future__ import annotations
import argparse,hashlib,json,random,sys
from collections import Counter,defaultdict
from pathlib import Path
import torch
import sentencepiece as spm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from train_amn_tm_v3 import read_split,prepare_rows,choose_distinct,multi_turn_pair
from amn_tm_v3 import INTENTS
from amn_tm_v31 import restore

def qtokens(sp,text,lim):
    return min(len(sp.encode(text)),lim-1)+1

def replay(rows,sp,metrics):
    rng=random.Random(73+200)
    counts=Counter()
    stage_counts=defaultdict(Counter)
    lm=inputs=intents=ground=dialog=0
    for stage,steps in metrics["scheduled_stages"].items():
        for i in range(steps):
            if stage=="answer_language_model":
                sample=choose_distinct(rng,rows,4)
                for r in sample:
                    x,y=r["lm"];lm+=sum(t!=-100 for t in y);inputs+=len(x)
                    stage_counts[stage][r["prompt"]]+=1
            elif stage=="weak_intent":
                sample=choose_distinct(rng,rows,4)
                intents+=len(sample)
                for r in sample:
                    inputs+=qtokens(sp,r["prompt"],64)
                    stage_counts[stage][r["prompt"]]+=1
            elif stage=="in_batch_grounding":
                sample=choose_distinct(rng,rows,8)
                ground+=len(sample)
                for r in sample:
                    inputs+=qtokens(sp,r["prompt"],64)+qtokens(sp,r["answer"],64)
                    stage_counts[stage][r["prompt"]]+=1
            elif stage=="synthetic_multiturn":
                for _ in range(4):
                    for attempt in range(20):
                        first,second=choose_distinct(rng,rows,2)
                        item=multi_turn_pair(sp,first,second,192)
                        if item is not None:break
                    else:item=second["sft"]
                    x,y=item
                    lm+=sum(t!=-100 for t in y);inputs+=len(x);dialog+=1
                    stage_counts[stage][second["prompt"]]+=1
            elif stage=="answer_sft_with_replay":
                sample=choose_distinct(rng,rows,4)
                for r in sample:
                    x,y=r["sft"]
                    lm+=sum(t!=-100 for t in y);inputs+=len(x)
                    counts[r["prompt"]]+=1
                    stage_counts[stage][r["prompt"]]+=1
                if i%4==0:
                    intents+=len(sample);ground+=len(sample)
                    for r in sample:
                        inputs+=qtokens(sp,r["prompt"],64)+qtokens(sp,r["answer"],64)
            else:raise RuntimeError("Unknown training stage: "+stage)
    actual={"train_lm_target_tokens":lm,"train_input_tokens":inputs,
            "weak_intent_supervision_examples":intents,
            "contrastive_positive_pairs":ground,
            "synthetic_multiturn_examples":dialog}
    for key,v in actual.items():
        if v!=metrics[key]:
            raise RuntimeError("Training replay mismatch "+key+
                               " observed="+str(v)+" expected="+str(metrics[key]))
    return counts,stage_counts,actual

def select(rows,counts,limit=12):
    by_prompt={r["prompt"]:r for r in rows}
    groups=defaultdict(list)
    for prompt,number in counts.items():
        row=by_prompt[prompt]
        answer=row["answer"]
        if not (10<=len(prompt)<=115 and 20<=len(answer)<=350):continue
        if "\n" in prompt or "\n" in answer:continue
        if "http" in prompt or any(ch in prompt for ch in "[]{}<>"):continue
        category=INTENTS[row["intent"]]
        key=(-number,abs(len(prompt)-45),abs(len(answer)-100),prompt)
        groups[category].append((key,row,number))
    for group in groups.values():group.sort(key=lambda x:x[0])
    chosen=[];seen=set()
    def add(row,n):
        k=" ".join(row["prompt"].casefold().split())
        if k in seen:return
        chosen.append((row,n,INTENTS[row["intent"]]));seen.add(k)
    for category in ("explanation","procedure","comparison","calculation","writing","other"):
        for _,row,n in groups[category][:2]:add(row,n)
    if len(chosen)<limit:
        remaining=sorted((entry for group in groups.values() for entry in group),
                         key=lambda x:x[0])
        for _,row,n in remaining:
            add(row,n)
            if len(chosen)>=limit:break
    return chosen[:limit]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/open_turkish_v31")
    parser.add_argument("--results",default="results/amn_tm_v31")
    args=parser.parse_args()
    data=Path(args.data);folder=Path(args.results)
    metrics=json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    if manifest!=metrics["data"]:
        raise RuntimeError("Source data manifest does not match training")
    tokenizer_sha=hashlib.sha256((folder/"tokenizer.model").read_bytes()).hexdigest()
    if tokenizer_sha!=metrics["tokenizer_sha256"]:
        raise RuntimeError("Trained tokenizer hash mismatch")
    sp=spm.SentencePieceProcessor(model_file=str(folder/"tokenizer.model"))
    rows=prepare_rows(sp,read_split(data,"train"),
                      metrics["effective_training_context_max"])
    if len(rows)!=metrics["data_counts"]["train"]:
        raise RuntimeError("Encoded training row count mismatch")
    counts,stages,reconstruction=replay(rows,sp,metrics)
    chosen=select(rows,counts)
    if len(chosen)<6:raise RuntimeError("Too few verified short questions")
    torch.set_num_threads(2)
    model,metadata=restore(folder/"best.pt")
    model.eval()
    responses=[]
    for i,(row,occurrences,category) in enumerate(chosen):
        result=model.generate(sp,row["prompt"],max_new_tokens=120,
                              temperature=0.7,top_k=32,seed=873+i)
        x,y=row["sft"]
        with torch.inference_mode():
            logits=model(torch.tensor([x]))
            expected=torch.tensor(y)
            nll=torch.nn.functional.cross_entropy(logits[0],expected,
                                                   ignore_index=-100).item()
        item={"question":row["prompt"],"training_reference":row["answer"],
              "weak_intent_class":category,"sft_exposures":occurrences,
              "reference_teacher_forced_nll":nll,
              "model_answer":result["text"],"generated_tokens":result["tokens"],
              "eos":result["eos"]}
        responses.append(item)
        print("SEEN_QUESTION",json.dumps(item,ensure_ascii=False),flush=True)
    audit={"source":manifest,"model":"AMNTMv3.1",
           "training_step":metadata.get("global_step"),
           "original_training_sampling_replayed":True,
           "training_exposure_counters_verified":reconstruction,
           "tokenizer_sha256":tokenizer_sha,
           "encoded_train_examples":len(rows),
           "different_questions_really_used_in_sft":len(counts),
           "number_of_sft_examples_seen":sum(counts.values()),
           "intent_distribution_of_train_pool":dict(Counter(
               INTENTS[r["intent"]] for r in rows)),
           "selection_before_inference":True,
           "note":"This tests memorization of seen training QAs, not heldout generalization.",
           "results":responses}
    output=folder/"seen_training_questions"
    output.mkdir(parents=True,exist_ok=True)
    (output/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),
                                     encoding="utf-8")
    lines=["# AMN-TM v3.1 — Gerçek eğitimde GÖRÜLMÜŞ sorular","",
           "Yalnız eğitim sırasında cevap SFT güncellemesi için gerçekten seçilen",
           "sorular kullanıldı. Kaynak SHA, tokenizer SHA ve beş sayısal eğitim",
           "sayacı yeniden hesaplanarak birebir doğrulandı.","",
           "Bu veri ezberleme / görülmüş soru testidir; genelleme ölçümü değildir.",
           "Model cevapları olduğu gibi kaydedildi, düzenlenmedi.","",
           "Eğitim havuzu: "+str(len(rows))+
           "; SGD'de SFT için seçilen farklı soru: "+str(len(counts))+
           "; toplam SFT seçimi: "+str(sum(counts.values())),""]
    for i,r in enumerate(responses,1):
        lines.extend(["## Soru "+str(i), "**Eğitimdeki soru:** "+r["question"],"",
                      "**Veri kümesindeki doğru/hedef cevap:** "+r["training_reference"],"",
                      "**AMN'nin gerçek ürettiği cevap:** "+(r["model_answer"] or "(boş)"),"",
                      "Eğitim SFT seçim sayısı: "+str(r["sft_exposures"])+
                      "; otomatik niyet: "+r["weak_intent_class"]+
                      "; referans NLL: "+f'{r["reference_teacher_forced_nll"]:.4f}',""])
    (output/"seen_qa_tr.md").write_text("\n".join(lines),encoding="utf-8")
    print("DONE",json.dumps({"replay_passed":True,"questions":len(responses),
          "unique_sft_seen":len(counts)},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
