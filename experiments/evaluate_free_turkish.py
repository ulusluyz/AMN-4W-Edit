"""Blind(ish) end-to-end raw Turkish evaluation for experimental AMN pipeline.

No reference or gold triples are passed to inference. Reports total accuracy
INCLUDING refusals and semantic-parser mistakes. The test split uses held-out
fact BINDINGS, not necessarily completely unseen lexical/grammatical patterns.
The generated corpus and grammar were designed during model development, so
this is NOT an independent real-world blind Turkish benchmark.
"""
import argparse
import json
from pathlib import Path

import torch

from relational_qa import VOCAB
from free_turkish_core import (
    ROOT, STYLES, infer, sample_raw, reference_text,
)

@torch.no_grad()
def evaluate(records, parser, base, decoder, threshold=.67, detailed_limit=16):
    total=len(records);covered=0;correct=0;parse_correct=0
    responses=[];why={};correct_by_style={x:[0,0] for x in STYLES}
    for ix, (sentences,question,triples,query,answer) in enumerate(records):
        style=STYLES[ix%len(STYLES)]
        context=' '.join(sentences)
        subj,rel=(VOCAB[int(x)] for x in query)
        expected=reference_text(subj,rel,VOCAB[int(answer)],style)
        try:
            prediction=infer(context,question,style,parser,base,decoder,threshold)
            covered+=1
            is_parse=prediction['predicted_query']==[subj,rel] and prediction['predicted_facts']==[
                [VOCAB[int(z)] for z in trip] for trip in triples]
            parse_correct+=int(is_parse)
            output=prediction['generated'];confidence=prediction['confidence_min']
            is_correct=(output==expected)
            correct+=int(is_correct)
            correct_by_style[style][0]+=int(is_correct)
            if len(responses)<detailed_limit or (not is_correct and sum(not x.get('exact',True) for x in responses)<4):
                responses.append({'raw_context':context,'raw_question':question,'expected':expected,
                     'generated':output,'exact':is_correct,'parsed_correctly':is_parse,
                     'confidence_min':confidence,'style':style})
        except ValueError as ex:
            cause=str(ex);why[cause]=why.get(cause,0)+1
            if len(responses)<detailed_limit:
                responses.append({'raw_context':context,'raw_question':question,'expected':expected,
                    'generated':None,'exact':False,'rejected':cause,'style':style})
        correct_by_style[style][1]+=1
    return {'total':total,'accepted':covered,'coverage':covered/total,
        'exact_generated_count':correct,'end_to_end_exact_accuracy':correct/total,
        'parse_episodes_exact_within_accepted':parse_correct,
        'answer_accuracy_conditional_accept':correct/max(1,covered),
        'styles':{k:{'correct':v[0],'total':v[1]} for k,v in correct_by_style.items()},
        'rejections':why,'examples':responses}

@torch.no_grad()
def ood_probes(parser,base,decoder):
    samples=[
      ('Kuantum fiziği atom altı parçacıkları inceler.', 'Bir kara deliğin nasıl oluştuğunu anlatır mısın?'),
      ('Bora dün yeni bir telefon satın aldı.', 'Bora hangi marka telefonu aldı?'),
      ('Ali şu anda evde.', 'Ali neden sinirli?'),
      ('Yarın hava yağmurlu olacak.', 'Yağmurun oluşumunu ayrıntılı açıklar mısın?'),
    ]
    result=[]
    for context,question in samples:
        try:
            output=infer(context,question,'detayli',parser,base,decoder,.67)
            result.append({'context':context,'question':question,
                'status':'UNSAFE_ACCEPT_OOD','model_response':output['generated'],
                'min_confidence':output['confidence_min']})
        except ValueError as error:
            result.append({'context':context,'question':question,
                'status':'rejected','reason':str(error)})
    return result


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--output',default='results/free_turkish_tagged')

    cli.add_argument('--count',type=int,default=300)
    cli.add_argument('--threshold',type=float,default=.55)
    cli.add_argument('--seed',type=int,default=69021)
    args=cli.parse_args()
    torch.set_num_threads(1)
    folder=ROOT/args.output
    from turkish_tagged_parser import load_pipeline
    parser,base,decoder=load_pipeline(folder)
    seen=sample_raw('test',args.count,args.seed,holdout_forms=False)
    novel=sample_raw('test',args.count,args.seed+1,holdout_forms=True)
    results={
      'protocol':{'independent_test_triples':31,'test_samples_per_condition':args.count,
        'trained_semantic_extraction':'byte-level bidirectional GRU, no entity regex',
        'frozen_memory':'Dual Upsert AMN 50-step checkpoint',
        'output':'autoregressive GRU word tokens, no formatting at inference',
        'limit':'synthetic finite person/location/colour domain; not open-domain Turkish'},
      'known_phrasing':evaluate(seen,parser,base,decoder,args.threshold),
      'unseen_phrasing':evaluate(novel,parser,base,decoder,args.threshold),
      'out_of_domain':ood_probes(parser,base,decoder),
    }
    path=folder/'end_to_end_validation.json'
    path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[
      '# Ham Türkçe girişten modelin ürettiği uzun cevaplar',
      '',
      'Girdi metinleri kural tabanlı kişi/ilişki/değer ayrıştırıcısına verilmez; öğrenilmiş byte-GRU kullanılır.',
      'Çıkış kelimeleri GRU tarafından seçilir. **Eğitim hedefleri sınırlı elle yazılmış cümle tiplerinden oluşur.**',
      'Bu bir genel Türkçe LLM veya sınırsız serbest metin üretimi değildir.',
      '',
      '## Sonuçlar','',
      '| Test | Kabul | Tam doğru uzun cevap | Kapsam |',
      '|---|---:|---:|---:|',
    ]
    for title,key in [('Bilinen ifade türleri','known_phrasing'),('Eğitimde görülmemiş ifade kalıpları','unseen_phrasing')]:
        r=results[key]
        lines.append(f'| {title} | {r["accepted"]}/{r["total"]} | {r["exact_generated_count"]}/{r["total"]} | %{r["end_to_end_exact_accuracy"]*100:.2f} |')
    for title,key in [('Bilinen cümle tipleri','known_phrasing'),('Yeni cümle kalıpları','unseen_phrasing')]:
        lines.extend(['',f'## {title}',''])
        for i,e in enumerate(results[key]['examples'][:12]):
            lines.extend([f'### {i+1}. örnek',
             '**Ham bilgi:** '+e['raw_context'],'**Ham soru:** '+e['raw_question'],
             '**Gerçek model çıktısı:** '+(e['generated'] or '(reddedildi)'),
             '**Beklenen:** '+e['expected'],
             '**Durum:** '+('DOĞRU' if e['exact'] else 'YANLIŞ / RED'), ''])
    lines.extend(['','## Alan dışı sorular',''])
    for case in results['out_of_domain']:
        lines.extend(['**Soru:** '+case['question'],
                      '**Durum:** '+case['status'],
                      '**Yanıt:** '+case.get('model_response',case.get('reason','')),''])
    (folder/'end_to_end_examples_tr.md').write_text('\n'.join(lines),encoding='utf-8')
    print('VALIDATION',json.dumps({key:{k:v for k,v in results[key].items() if k not in ('examples','rejections')}
          for key in ('known_phrasing','unseen_phrasing')},ensure_ascii=False),flush=True)
    print('OOD',json.dumps(results['out_of_domain'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()