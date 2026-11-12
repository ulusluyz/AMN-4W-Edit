"""AMN grounded memory v1 -- *local* source-verified question/fact routing.

Uses only already-reviewed repository TRAIN split facts. Never downloads anything
and NEVER reads validation or test split into the answering memory.

Two mechanisms:
1) High-precision Turkish topic/question routing (hard safety gate).
2) Optional neural AMN query/evidence embeddings to rank ambiguous matches.
A selected fact is returned VERBATIM with its source; this is EXTRACTIVE MEMORY,
not an invented claim that the autoregressive decoder learned it.

No wrong-topic fallback, no repeating-word loop, no fake 'Web...' outputs.
Unknown or off-task questions return an explicit non-answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path

import torch
from torch.nn import functional as F
import sentencepiece as spm

from amn_tm_v3 import collate_questions
from amn_tm_v31 import restore

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_TRAIN=ROOT/"data/amn_world_v1/train.jsonl"
DEFAULT_WEIGHTS=ROOT/"results/amn_world_v1_train/best.pt"
DEFAULT_TOKENIZER=ROOT/"results/amn_tm_v31/tokenizer.model"

NOT_FOUND="Bu soruya ilişkin doğrulanmış bilgi mevcut belleğimde bulunmuyor."

def tr_lower(text):
    return str(text).replace("İ","i").replace("I","ı").lower()

def normalize(text):
    text=tr_lower(text)
    text=re.sub(r"[^\wçğıöşü]+"," ",text,flags=re.UNICODE)
    return " ".join(text.split())

def text_tokens(text):
    return normalize(text).split()

# Only DEFINITIONS and WHO-IS queries are supported from encyclopedia leads.
# A definition does NOT imply that the model knows how to perform an action.
DEFINITION_SUFFIXES=(
    "nedir","ne demektir","kimdir","kimdi",
    "hakkında kısa bilgi verir misin","hakkında kısa bilgi ver",
    "hakkında bilgi verir misin","hakkında bilgi ver",
    "kime denir","tanımı nedir","anlamı nedir",
)
DEFINITION_SUFFIXES=tuple(sorted(DEFINITION_SUFFIXES,key=len,reverse=True))
ALLOWED_TITLE_PROMPTS=(
    "{topic} nedir","{topic} ne demektir","{topic} kimdir",
    "{topic} kime denir","{topic} hakkında bilgi verir misin",
    "{topic} hakkında kısa bilgi verir misin",
)

@dataclass(frozen=True)
class MemoryResult:
    question:str
    answer:str
    source:str
    source_url:str
    topic:str
    match_type:str
    confidence:float
    neural_similarity:float|None=None
    @property
    def found(self):return bool(self.topic)
    def as_dict(self):
        return {"question":self.question,"answer":self.answer,
                "source":self.source,"source_url":self.source_url,
                "topic":self.topic,"method":self.match_type,
                "confidence":self.confidence,
                "neural_similarity":self.neural_similarity,
                "model_generated":False}

class MemoryIndex:
    def __init__(self,train_path=DEFAULT_TRAIN,model=None,tokenizer=None):
        self.path=Path(train_path)
        raw=self.path.read_bytes()
        self.source_sha256=hashlib.sha256(raw).hexdigest()
        records=[json.loads(line) for line in raw.decode("utf-8").splitlines()
                 if line.strip()]
        self.records=records
        self.model=model
        self.tokenizer=tokenizer
        self.by_question={}
        self.by_topic=defaultdict(list)
        self.definition_topics={}
        for row in records:
            if not row.get("source_url") or not row.get("answer"):
                raise ValueError("Invalid or un-attributed train evidence")
            prompt=normalize(row["prompt"])
            if prompt in self.by_question and self.by_question[prompt]["answer"]!=row["answer"]:
                raise ValueError("Conflicting answers to the same memory question")
            self.by_question[prompt]=row
            t=normalize(row["topic"])
            self.by_topic[t].append(row)
            if row["category"] in ("resmi_isimler",):
                continue
            # Definitions from first-sentence trusted source. Select one
            # deterministic canonical reference even with 3 question forms.
            if t not in self.definition_topics:
                self.definition_topics[t]=row
            elif row["answer"]!=self.definition_topics[t]["answer"]:
                raise ValueError("Inconsistent definition for the same topic "+t)
        self.indexed_topics=len(self.by_topic)
        if model is not None and tokenizer is None:
            raise ValueError("Neural ranker requires tokenizer")

    @staticmethod
    def _subject(normalized_question):
        # Keep natural Turkish word-order, never substitute unrelated titles.
        q=normalized_question
        for suffix in DEFINITION_SUFFIXES:
            if q.endswith(" "+suffix):
                return q[:-(len(suffix)+1)].strip()
        if q.startswith("tanımla "):
            return q[len("tanımla "):].strip()
        if q.startswith("açıkla "):
            return q[len("açıkla "):].strip()
        return None

    def _candidate_rows(self,question):
        q=normalize(question)
        if not q:
            return [],"empty"
        if q in self.by_question:
            return [self.by_question[q]],"exact_train_question"
        subject=self._subject(q)
        if subject is None:
            return [],"unsupported_intent"
        # Don't answer generic 'bir şey...' with coincidental topic tokens.
        if not subject or len(subject)<2:
            return [],"unknown_topic"
        if subject in self.definition_topics:
            return [self.definition_topics[subject]],"exact_topic"
        # The article titles may carry a parenthetical qualifier.
        # Do not infer from a single overlapping token (e.g. Güneş Sistemi
        # cannot be returned as the answer to 'Güneş nedir').
        return [],"unknown_topic"

    @torch.inference_mode()
    def _score_neural(self,question,rows):
        if self.model is None:return None
        sp=self.tokenizer;model=self.model
        model.eval()
        q=collate_questions(sp,[question],max_len=72)
        a=collate_questions(sp,[x["answer"] for x in rows],max_len=92)
        # Retrieve using the existing AMN query and evidence heads.
        qs=F.normalize(model.query_head(model.pooled(q)),dim=-1)
        ev=F.normalize(model.evidence_head(model.pooled(a)),dim=-1)
        return (qs @ ev.T)[0].detach().cpu().tolist()

    def answer(self,question):
        rows,mode=self._candidate_rows(question)
        if not rows:
            return MemoryResult(question,NOT_FOUND,"","","",mode,0.0)
        neural=self._score_neural(question,rows)
        if len(rows)>1:
            # Neural head is a ranker among lexical/gated candidates only.
            # It can NEVER invent or select an unrelated off-topic source.
            best=max(range(len(rows)),key=lambda i:neural[i]) if neural else 0
        else:
            best=0
        row=rows[best]
        score=neural[best] if neural else None
        return MemoryResult(question,row["answer"],row["source"],
                            row["source_url"],row["topic"],mode,1.0,score)

def load_memory(train=DEFAULT_TRAIN,weights=DEFAULT_WEIGHTS,tokenizer=DEFAULT_TOKENIZER,neural=True):
    if neural:
        model,_=restore(weights)
        model.eval()
        sp=spm.SentencePieceProcessor(model_file=str(tokenizer))
        if sp.vocab_size()!=model.config["vocab_size"]:
            raise RuntimeError("Tokenizer checkpoint mismatch")
        return MemoryIndex(train_path=train,model=model,tokenizer=sp)
    return MemoryIndex(train_path=train)

def ask_many(questions,train=DEFAULT_TRAIN,weights=DEFAULT_WEIGHTS,
             tokenizer=DEFAULT_TOKENIZER,neural=True):
    mem=load_memory(train,weights,tokenizer,neural=neural)
    return [mem.answer(q).as_dict() for q in questions]
