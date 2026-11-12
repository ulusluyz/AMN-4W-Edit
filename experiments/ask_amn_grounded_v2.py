"""Offline CLI to ask the AMN source-grounded factual memory.

Examples:
    python experiments/ask_amn_grounded_v2.py "Kedi nedir?"
    python experiments/ask_amn_grounded_v2.py "Salata nasıl yapılır?"

Returns retrieved source evidence, never arbitrary decoder text.
--inspect-raw also shows the neural decoder independently; that debug text
is NEVER silently presented as a verified answer.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from amn_grounded_memory_v1 import load_memory

def main():
    p=argparse.ArgumentParser()
    p.add_argument("question",nargs="*",help="Question text")
    p.add_argument("--model",default="results/amn_world_v2_grounded/best.pt")
    p.add_argument("--tokenizer",default="results/amn_tm_v31/tokenizer.model")
    p.add_argument("--train",default="data/amn_world_v1/train.jsonl")
    p.add_argument("--inspect-raw",action="store_true")
    p.add_argument("--output",default="")
    args=p.parse_args()
    questions=[" ".join(args.question)] if args.question else [
        "Kedi nedir?","Köpek nedir?","Denklem nedir?",
        "Cumhuriyet nedir?","Salata nasıl yapılır?",
        "20 tane erkek insan 8 tane kadın insan ismi yaz"
    ]
    memory=load_memory(train=args.train,weights=args.model,
                       tokenizer=args.tokenizer,neural=True)
    rows=[]
    for q in questions:
        item=memory.answer(q).as_dict()
        if args.inspect_raw:
            output=memory.model.generate(memory.tokenizer,q,
                  max_new_tokens=100,temperature=0,top_k=0)
            item["unverified_raw_decoder"]=output["text"]
            item["raw_decoder_is_verified"]=False
        rows.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":main()
