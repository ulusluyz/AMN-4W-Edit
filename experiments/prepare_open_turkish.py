"""Prepare truly Turkish instruction/response text from an Apache-2.0 corpus.

Original source: https://huggingface.co/datasets/TFLai/Turkish-Alpaca
Only real source examples are used. No generated templates or slot labels.

Usage:
    python experiments/prepare_open_turkish.py --output data/open_turkish
    python experiments/prepare_open_turkish.py --source path/to/data.json --output data/open_turkish

The source data is not committed; only the SHA-256 manifest is archived.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time, urllib.request, urllib.parse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE='TFLai/Turkish-Alpaca'
URL='https://huggingface.co/datasets/TFLai/Turkish-Alpaca/resolve/main/data.json?download=true'
ROWS='https://datasets-server.huggingface.co/rows'
LICENSE='Apache-2.0'


def fetch_source(target):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size>100000:
        return target
    error=None
    for retry in range(4):
        try:
            request=urllib.request.Request(URL,headers={'User-Agent':'AMN-4W-Edit-research/1.0','Accept':'application/json'})
            with urllib.request.urlopen(request,timeout=120) as conn, target.open('wb') as out:
                import shutil
                shutil.copyfileobj(conn,out)
            if target.stat().st_size<100000:raise RuntimeError('Dataset too small')
            return target
        except Exception as ex:
            error=ex
            target.unlink(missing_ok=True)
            time.sleep(2*(retry+1))
    print('Direct dataset download unavailable; trying public read-only row API:',str(error),flush=True)
    # Pages API fallback; fetch thousands of real, verifiable dataset rows.
    all_rows=[]
    for offset in range(0,20000,100):
        params=urllib.parse.urlencode({'dataset':SOURCE,'config':'default','split':'train',
                                       'offset':offset,'length':100})
        with urllib.request.urlopen(ROWS+'?'+params,timeout=60) as res:
            data=json.load(res)
        rows=data.get('rows',[])
        if not rows:break
        all_rows.extend(row['row'] for row in rows)
        if offset % 2000==0:print('ROWS',len(all_rows),flush=True)
    if len(all_rows)<500:raise RuntimeError('Public dataset download failed and row API returned too few rows')
    target.write_text(json.dumps(all_rows,ensure_ascii=False),encoding='utf-8')
    return target


def unpack_source(filepath):
    data=json.loads(Path(filepath).read_text(encoding='utf-8'))
    if isinstance(data,dict):
        if 'train' in data:data=data['train']
        elif 'data' in data:data=data['data']
    if not isinstance(data,list):raise ValueError('Dataset must contain JSON list')
    for row in data:
        if not isinstance(row,dict):continue
        instr=row.get('instruction') or row.get('prompt') or ''
        addon=row.get('input') or ''
        answer=row.get('output') or row.get('response') or ''
        if not isinstance(instr,str) or not isinstance(addon,str) or not isinstance(answer,str):continue
        prompt=(instr.strip()+('\n'+addon.strip() if addon.strip() else '')).strip()
        answer=answer.strip()
        if prompt and answer:
            yield {'prompt':prompt,'answer':answer}


def split_rows(source_file,max_seq_len=384,max_samples=16000,min_answer_bytes=48):
    from amn_open_turkish import make_training_example,make_prefix
    unique={}; seen_hashes=set(); skipped={'empty':0,'short':0,'too_long':0,'duplicate':0}
    records=list(unpack_source(source_file))
    for record in records:
        prompt,answer=record['prompt'],record['answer']
        # Do not treat newlines / arbitrary text as structured entity slots.
        prompt_key=' '.join(prompt.casefold().split())
        if prompt_key in unique:
            skipped['duplicate']+=1;continue
        if len(answer.encode('utf-8'))<min_answer_bytes:
            skipped['short']+=1;continue
        if make_training_example(prompt,answer,max_seq_len) is None:
            skipped['too_long']+=1;continue
        # If answer appears verbatim in prompt, ignore to avoid copy leakage.
        if answer.casefold() in prompt.casefold():
            skipped['duplicate']+=1;continue
        unique[prompt_key]=record
    # Stable hashed prompt grouping prevents identical-input split leakage.
    eligible=sorted(unique.items(),key=lambda kv:hashlib.sha256(('select|'+kv[0]).encode()).digest())[:max_samples]
    partitions={'train':[],'val':[],'test':[]}
    for key,record in eligible:
        h=int.from_bytes(hashlib.sha256(('split|'+key).encode()).digest()[:8],'big')%1000
        part='test' if h<70 else 'val' if h<140 else 'train'
        partitions[part].append(record)
    return partitions,{'source_rows':len(records),'eligible':len(unique),'selected':len(eligible),'skipped':skipped,
                       'train':len(partitions['train']),'val':len(partitions['val']),'test':len(partitions['test']),
                       'min_answer_bytes':min_answer_bytes,'max_seq_len':max_seq_len,'max_samples':max_samples}


def run(args):
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    source=Path(args.source) if args.source else fetch_source(folder/'raw_data.json')
    parts,stats=split_rows(source,args.max_seq_len,args.max_samples,args.min_answer_bytes)
    if stats['train']<args.min_train:
        raise RuntimeError(f'Too few suitable real Turkish examples: {stats}; min_train={args.min_train}')
    if not parts['val'] or not parts['test']:
        raise RuntimeError('Empty validation or test split')
    for split,rows in parts.items():
        (folder/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest={'dataset':SOURCE,'dataset_page':f'https://huggingface.co/datasets/{SOURCE}',
              'license':LICENSE,'source_file_sha256':sha, 'split_key':'normalized prompt',
              'split_protocol':'SHA256 deterministic; 86% train, 7% dev, 7% test, exact input dedup',
              'note':'SFT real human-readable Turkish text; no template or structured slot labels',**stats}
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('MANIFEST',json.dumps(manifest,ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default=None)
    p.add_argument('--output',default='data/open_turkish')
    p.add_argument('--max-seq-len',type=int,default=384)
    p.add_argument('--max-samples',type=int,default=16000)
    p.add_argument('--min-answer-bytes',type=int,default=48)
    p.add_argument('--min-train',type=int,default=2500)
    run(p.parse_args())

if __name__=='__main__':main()
