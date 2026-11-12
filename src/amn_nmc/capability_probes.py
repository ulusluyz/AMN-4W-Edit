"""Zero-training diagnostic probes of in-context memory and 2-hop reasoning.

These are held-out synthetic *evaluation prompts*, never a training corpus.
A score is not evidence of general reasoning unless it exceeds chance robustly
across unseen cases and multiple independently trained seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .config import NMCConfig
from .model import AMNNMC
from .train import _atomic_json


def challenge_set():
    distraction = ' Bu paragraf, yalnızca bağlam uzaklığını ölçmek için eklendi.' * 8
    return [
        {'class':'near_fact','prompt':'Deney kaydında yeşil sandığın kodu A71 olarak yazıldı. Yeşil sandığın kodu:',
         'correct':' A71','wrong':' B82'},
        {'class':'far_fact','prompt':'Deney kaydında yeşil sandığın kodu A71 olarak yazıldı.'+distraction+' Yeşil sandığın kodu:',
         'correct':' A71','wrong':' B82'},
        {'class':'two_entities','prompt':'Kırmızı kabın kodu M14, mavi kabın kodu K26 olarak belirtildi. Mavi kabın kodu:',
         'correct':' K26','wrong':' M14'},
        {'class':'overwrite','prompt':'Birinci kutunun kodu C35 idi. Daha sonra kod D46 olarak değiştirildi. Birinci kutunun yeni kodu:',
         'correct':' D46','wrong':' C35'},
        {'class':'two_hop','prompt':'Sera kutusu Bölme A içindedir. Bölme A, Oda 7 içindedir. Sera kutusu hangi odada?',
         'correct':' 7','wrong':' 2'},
        {'class':'nested_owner','prompt':'Anahtar Efe tarafından Bora kişisine verildi. Daha sonra Bora anahtarı Cem kişisine verdi. Anahtar şimdi kimde?',
         'correct':' Cem','wrong':' Efe'},
    ] + [
        # Counterfactual controls prevent a preferred answer string from
        # masquerading as actual associative recall or reasoning.
        {'class':'near_fact_swapped','prompt':'Deney kaydında yeşil sandığın kodu B82 olarak yazıldı. Yeşil sandığın kodu:',
         'correct':' B82','wrong':' A71'},
        {'class':'far_fact_swapped','prompt':'Deney kaydında yeşil sandığın kodu B82 olarak yazıldı.'+distraction+' Yeşil sandığın kodu:',
         'correct':' B82','wrong':' A71'},
        {'class':'two_entities_swapped','prompt':'Kırmızı kabın kodu K26, mavi kabın kodu M14 olarak belirtildi. Mavi kabın kodu:',
         'correct':' M14','wrong':' K26'},
        {'class':'overwrite_swapped','prompt':'Birinci kutunun kodu D46 idi. Daha sonra kod C35 olarak değiştirildi. Birinci kutunun yeni kodu:',
         'correct':' C35','wrong':' D46'},
        {'class':'two_hop_swapped','prompt':'Sera kutusu Bölme A içindedir. Bölme A, Oda 2 içindedir. Sera kutusu hangi odada?',
         'correct':' 2','wrong':' 7'},
        {'class':'nested_owner_swapped','prompt':'Anahtar Cem tarafından Bora kişisine verildi. Daha sonra Bora anahtarı Efe kişisine verdi. Anahtar şimdi kimde?',
         'correct':' Efe','wrong':' Cem'},
    ]


@torch.inference_mode()
def score_candidates(model, prefix, options):
    model.eval()
    prefix_bytes = prefix.encode('utf-8')
    state = None
    logits = None
    for start in range(0,len(prefix_bytes),128):
        inp=torch.tensor([list(prefix_bytes[start:start+128])],dtype=torch.long)
        logits,state=model(inp,state)
    nll = {}
    for option in options:
        score=0.0
        next_logits=logits
        option_state=state
        for b in option.encode('utf-8'):
            score -= float(F.log_softmax(next_logits[0,-1].float(),dim=-1)[b])
            next_logits,option_state=model(torch.tensor([[b]],dtype=torch.long),option_state)
        nll[option]=score/len(option.encode('utf-8'))
    return nll


def probe(model, *, max_cases=None):
    cases=challenge_set()[:max_cases]
    outcomes=[]
    for case in cases:
        scores=score_candidates(model,case['prompt'],[case['correct'],case['wrong']])
        choice=min(scores,key=scores.get)
        outcomes.append({'class':case['class'],'correct':case['correct'],
                         'prediction':choice,'correct_rank_1':choice==case['correct'],
                         'score_correct':scores[case['correct']],
                         'score_wrong':scores[case['wrong']],
                         'margin_nats_per_byte':scores[case['wrong']]-scores[case['correct']]})
    accuracy=sum(v['correct_rank_1'] for v in outcomes)/len(outcomes)
    paired_success=(sum(outcomes[i]['correct_rank_1'] and outcomes[i+6]['correct_rank_1']
                        for i in range(6)) if len(outcomes)==12 else None)
    return {'total':len(outcomes),'correct':sum(v['correct_rank_1'] for v in outcomes),
            'paired_success_count':paired_success, 'paired_total':(6 if paired_success is not None else None),
            'accuracy':accuracy,'chance_binary':0.5,'zero_training_from_probes':True,
            'interpretation':'diagnostic only; single binary item per class is NOT proof of general reasoning',
            'cases':outcomes}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=AMNNMC(NMCConfig(**payload['config']))
    model.load_state_dict(payload['model'])
    torch.set_num_threads(2)
    report=probe(model)
    report['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    _atomic_json(report,a.output)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
