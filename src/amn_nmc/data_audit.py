"""Conservative, auditable corpus triage for autoregressive language pretraining.

This does not certify grammatical/factual correctness, language identity or licensing.
No document is automatically marked approved for model training.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

PATTERNS = {
    'html_or_wiki_markup': re.compile(r'<(?:script|style|div|span|ref|br|table)\b|</?[a-z][^>]*>|\{\{|\}\}|\[\[(?!\d)',re.I),
    'web_navigation': re.compile(r'(?:çerez politikası|gizlilik politikası|üyelik sözleşmesi|tüm hakları saklıdır|cookie policy|all rights reserved|subscribe to)',re.I),
    'reference_sections': re.compile(r'(?im)^\s*(?:kaynakça|kaynaklar|ayrıca bakınız|dış bağlantılar|bibliyografya)\s*:?\s*$'),
    'inline_sections': re.compile(r'(?:\bKaynakça\s+Ayrıca bakınız\b|\bAyrıca bakınız\s+Kaynakça\b)',re.I),
    'email': re.compile(r'\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b',re.I),
    'iban': re.compile(r'\bTR\s*\d{2}(?:[\s-]*\d{4}){5}(?:[\s-]*\d{2})?\b',re.I),
    'control_or_invalid_unicode': re.compile('[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u202a-\u202e\ufffd\ud800-\udfff]'),
    'url_in_body': re.compile(r'https?://|www\.',re.I),
}
STOP = frozenset('ve bir bu için ile olarak olan da de daha çok en ise ancak gibi sonra önce kadar ya da veya tarafından arasında üzerindeki bazı farklı ayrıca böyle iki üç her hem ama çünkü bunu şu kendi diğer göre aynı tüm neden nasıl'.split())
TURKISH = set('çğıöşüÇĞİÖŞÜ')
REVIEW_ONLY = {'reference_sections','url_in_body','mixed_script','weak_turkish','low_sentence_density','repeated_lines'}

def normalize(text):
    return unicodedata.normalize('NFC',text.replace('\r\n','\n').replace('\r','\n').replace('\ufeff','')).strip()

def words(text):
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?",text,re.UNICODE)

def digest(text):
    return hashlib.sha256(' '.join(normalize(text).casefold().split()).encode('utf-8')).hexdigest()

def simhash(text):
    tokens=[w.casefold() for w in words(text)]
    if len(tokens)<25: return None
    grams=set(' '.join(tokens[i:i+5]) for i in range(len(tokens)-4))
    vote=[0]*64
    for gram in grams:
        v=int.from_bytes(hashlib.blake2b(gram.encode('utf-8'),digest_size=8).digest(),'big')
        for bit in range(64):
            vote[bit]+=1 if v>>bit&1 else -1
    return sum(1<<bit for bit,val in enumerate(vote) if val>0)

def inspect(text,min_chars=300,max_chars=250000):
    flags=[name for name,pattern in PATTERNS.items() if pattern.search(text)]
    n=len(text); toks=words(text); nw=len(toks)
    if n<min_chars or nw<35:flags.append('too_short')
    if n>max_chars:flags.append('too_long')
    if not text:flags.append('empty')
    letters=sum(c.isalpha() for c in text)
    digit=sum(c.isdigit() for c in text)/max(n,1)
    alpha=letters/max(n,1)
    punctuation=sum(unicodedata.category(c).startswith('P') for c in text)/max(n,1)
    foreign=sum(0x400<=ord(c)<=0x52f for c in text if c.isalpha())/max(letters,1)
    stops=sum(t.casefold() in STOP for t in toks)/max(nw,1)
    tr=sum(c in TURKISH for c in text)/max(letters,1)
    sentences=len(re.findall(r'[.!?](?:\s|$)',text))
    if digit>.20:flags.append('too_many_digits')
    if alpha<.47:flags.append('low_alphabetic_content')
    if punctuation>.24:flags.append('too_many_symbols')
    if foreign>.10:flags.append('mixed_script')
    # Only a heuristic, NOT reliable language identification.
    if nw>=35 and stops<.025 and tr<.008:flags.append('weak_turkish')
    if nw>=60 and sentences<2:flags.append('low_sentence_density')
    lines=[re.sub(r'\s+',' ',v).strip().casefold() for v in text.splitlines() if len(v.strip())>=12]
    if len(lines)>=6 and len(set(lines))/len(lines)<.75:flags.append('repeated_lines')
    if len(lines)>=6 and sum(bool(re.match(r'^(?:[-*•]|\d+[.)])\s',v)) for v in lines)/len(lines)>.6:
        flags.append('list_heavy')
    metrics={'chars':n,'bytes':len(text.encode('utf-8',errors='replace')),
             'words':nw,'sentences':sentences,'digit_ratio':round(digit,4),
             'alphabetic_ratio':round(alpha,4),'punctuation_ratio':round(punctuation,4),
             'cyrillic_ratio':round(foreign,4),'tr_stopword_ratio':round(stops,4),
             'tr_character_ratio':round(tr,4)}
    return metrics,sorted(set(flags))

def records(path):
    path=Path(path)
    files=sorted(path.rglob('*.jsonl'))+sorted(path.rglob('*.txt')) if path.is_dir() else [path]
    for file in files:
        if file.suffix=='.txt':
            try:
                yield {'text':file.read_text(encoding='utf-8',errors='strict'),'origin':str(file)}
            except UnicodeError:
                yield {'text':None,'origin':str(file),'source_error':'invalid_utf8'}
        elif file.suffix=='.jsonl':
            with file.open('rb') as stream:
                for i,raw in enumerate(stream,1):
                    origin=f'{file}:{i}'
                    try:
                        record=json.loads(raw.decode('utf-8','strict'))
                        if not isinstance(record,dict):raise ValueError('not object')
                        txt=record.get('text',record.get('extract'))
                        if not isinstance(txt,str):
                            yield {'text':None,'origin':origin,'source_error':'missing_raw_text'}
                        else:
                            yield {'text':txt,'origin':origin,
                                   'source':record.get('source',record.get('url','')),
                                   'title':record.get('title'),'category':record.get('category')}
                    except (UnicodeError,ValueError,json.JSONDecodeError):
                        yield {'text':None,'origin':origin,'source_error':'invalid_utf8_or_json'}
        else:
            raise ValueError('Only .jsonl and UTF-8 .txt files are supported')

def audit(source,output,*,declared_license=None,min_chars=300,max_chars=250000,max_rows=None,seed=17):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    seen=set();bands=defaultdict(list);counts=Counter();issues=Counter();domains=Counter();categories=Counter()
    clean=[];quarantine=[];review=[];n=0;total_bytes=0;candidate_bytes=0
    for row in records(source):
        if max_rows is not None and n>=max_rows:break
        n+=1;txt=row['text'];flags=[];metrics={};status='rejected'
        if txt is None:
            flags=[row.get('source_error','missing_raw_text')]
            key=hashlib.sha256(row['origin'].encode()).hexdigest()
        else:
            txt=normalize(txt)
            key=digest(txt);metrics,flags=inspect(txt,min_chars,max_chars)
            total_bytes+=metrics['bytes']
            if key in seen:flags.append('exact_duplicate')
            else:seen.add(key)
            sh=simhash(txt)
            if sh is not None and 'exact_duplicate' not in flags:
                near=False
                for shift in (0,16,32,48):
                    for other in bands[(shift,(sh>>shift)&65535)][:200]:
                        if (sh^other).bit_count()<=5:near=True;break
                    if near:break
                if near:flags.append('near_duplicate_simhash')
                else:
                    for shift in (0,16,32,48):
                        bands[(shift,(sh>>shift)&65535)].append(sh)
            src=row.get('source') or ''
            if not src:flags.append('missing_provenance')
            elif not str(src).startswith('https://'):flags.append('unverifiable_source_url')
            domain=urlparse(str(src)).hostname or 'unknown';domains[domain]+=1
            categories[row.get('category') or domain]+=1
            if not declared_license or declared_license=='UNSPECIFIED':flags.append('license_unverified')
            if not flags:
                status='candidate_only'
                clean.append({'text':txt,'source':src,'origin':row['origin'],'sha256':key,
                              'title':row.get('title'),'license_claim':declared_license})
                candidate_bytes+=metrics['bytes']
            elif set(flags).issubset(REVIEW_ONLY):status='needs_review'
        counts[status]+=1
        for f in set(flags):issues[f]+=1
        record={'origin':row['origin'],'sha256':key,'status':status,'flags':sorted(set(flags)),
                'metrics':metrics,'source':row.get('source'),'title':row.get('title')}
        if txt is not None:
            record['preview']='[withheld: possible PII]' if {'email','iban'}&set(flags) else txt[:170].replace('\n',' ')
        if status=='candidate_only':review.append(record)
        else:quarantine.append(record)
    random.Random(seed).shuffle(review)
    report={'source_input':str(source),'declared_license_not_independently_verified':declared_license,
            'total_rows':n,'status_counts':dict(counts),'flag_counts':dict(issues),
            'total_input_bytes':total_bytes,'candidate_bytes':candidate_bytes,
            'source_domains':dict(domains),'category_counts':dict(categories),
            'sample_for_human_review':review[:30],'candidate_is_approved_for_training':False,
            'quality_proof':'none: automatic flags cannot verify grammar, truth, source rights or human quality',
            'near_duplicate_limit':'SimHash 5-word shingles with <=5-bit LSH match; can miss close duplicates'}
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'candidate_only.jsonl').open('w',encoding='utf-8') as fp:
        for row in clean:fp.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (output/'rejected_or_review.jsonl').open('w',encoding='utf-8') as fp:
        for row in quarantine:fp.write(json.dumps(row,ensure_ascii=False)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description='Audit raw Turkish corpus, NEVER auto-approve training')
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--license')
    p.add_argument('--min-chars',type=int,default=300)
    p.add_argument('--max-rows',type=int)
    args=p.parse_args()
    result=audit(args.input,args.output,declared_license=args.license,
                 min_chars=args.min_chars,max_rows=args.max_rows)
    print(json.dumps({k:v for k,v in result.items() if k!='sample_for_human_review'},
                     ensure_ascii=False,indent=2))

if __name__=='__main__':main()
