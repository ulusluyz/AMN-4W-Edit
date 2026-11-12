"""Stage stratified, UNTRUNCATED Turkish Wikipedia records and audit them.

A sample audit is not a whole-corpus audit; no records are approved for training.
"""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from .data_audit import audit

DATASET='kaan39/turkish-wikipedia-dataset-clean'
DATASET_URL='https://huggingface.co/datasets/'+DATASET
API='https://datasets-server.huggingface.co/rows'
OFFSETS=(0,24000,48000,72000,96000,120000)

def sample_and_audit(folder,*,offsets=OFFSETS,length=24,fetch=None):
    if not 1<=length<=100 or any(n<0 for n in offsets):
        raise ValueError('invalid sampling parameters')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    raw=folder/'sampled_untruncated.jsonl'
    visited=set();count=0
    with raw.open('w',encoding='utf-8') as fp:
        for offset in offsets:
            query=urllib.parse.urlencode({'dataset':DATASET,'config':'default',
                                          'split':'train','offset':offset,'length':length})
            url=API+'?'+query
            if fetch is None:
                request=urllib.request.Request(url,headers={'User-Agent':'AMN-NMC-corpus-audit/0.1'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    obj=json.load(response)
            else:obj=fetch(url)
            if not isinstance(obj.get('rows'),list):
                raise ValueError('row API missing entries for offset '+str(offset))
            for entry in obj['rows']:
                r=entry['row']
                if not isinstance(r.get('text'),str) or not r['text']:continue
                identity=(r.get('page_id'),r.get('revision_id'))
                if identity in visited:continue
                visited.add(identity)
                if not str(r.get('source','')).startswith('https://tr.wikipedia.org/wiki/'):continue
                fp.write(json.dumps({'text':r['text'],'source':r['source'],
                                     'page_id':r.get('page_id'),
                                     'title':r.get('title'),
                                     'revision_id':r.get('revision_id'),
                                     'timestamp':str(r.get('timestamp',''))},ensure_ascii=False)+'\n')
                count+=1
    result=audit(raw,folder/'quality',declared_license='CC-BY-SA-4.0')
    provenance={'dataset':DATASET_URL,'dataset_id':DATASET,'api':API,
                'sample_offsets':list(offsets),'requested_per_offset':length,
                'sampled_articles':count,'dataset_revision_pinned':False,
                'entire_corpus_audited':False,'approved_for_training':False,
                'license_claim':'CC-BY-SA-4.0','quality_counts':result['status_counts']}
    (folder/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if count<10:raise RuntimeError(f'Insufficient actual source articles: {count}')
    return provenance

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--per-offset',type=int,default=24)
    args=p.parse_args()
    print(json.dumps(sample_and_audit(args.output,length=args.per_offset),
                     ensure_ascii=False,indent=2))

if __name__=='__main__':main()
