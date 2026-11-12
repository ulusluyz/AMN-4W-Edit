"""Curated Turkish Wikipedia world knowledge + official 2025 Turkish baby names.

Downloaded article first-sentences only, per-row source attribution, deterministic
topic-disjoint heldout splits, explicit rejects, SHA and quality audit.
Automatic screening does NOT replace expert factual review.
"""
import argparse,hashlib,html,json,re,time,random
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode,quote
from urllib.error import HTTPError,URLError

ROOT=Path(__file__).resolve().parents[1]
REST="https://tr.wikipedia.org/api/rest_v1/page/summary/"
ACTION="https://tr.wikipedia.org/w/api.php"
WIKI="https://tr.wikipedia.org"
TUIK="https://nip.tuik.gov.tr/?value=YeniDoganIsimleri"
AGENT="AMN-4W-Edit/1.1 (https://github.com/ulusluyz/AMN-4W-Edit; research)"
MEN=("Alparslan","Göktuğ","Metehan","Yusuf","Kerem","Miran","Ömer Asaf","Ömer",
     "Atlas","Aras","Miraç","Aslan","Kuzey","Poyraz","Alperen","Yiğit",
     "Eymen","Aybars","Ali Asaf","Çınar")
WOMEN=("Alya","Defne","Gökçe","Zeynep","Asel","Umay","Asya","Elisa","İnci","Duru","Elif")
BAD=("vikipedi","bu sayfa","bu madde","anlam ayrımı","aşağıda listelen",
     "şu anlamlara gelebilir","kaynak belirtilmeli","daha fazla bilgi",
     "ansiklopedi maddesi","[[","]]", "{{","}}")
SPLIT_SENT=re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])")

def digest(t):return hashlib.sha256(t.encode("utf-8")).hexdigest()
def norm(t):return " ".join(t.casefold().split())
def get_json(url):
    r=Request(url,headers={"User-Agent":AGENT,"Accept":"application/json"})
    with urlopen(r,timeout=22) as fp:
        b=fp.read(450000)
        if len(b)>=450000:raise ValueError("excessive API response")
        return json.loads(b)

def fetch(item):
    name=item["title"];cat=item["category"];errors=[]
    for endpoint in ("rest","action"):
        for attempt in range(2):
            try:
                if endpoint=="rest":
                    o=get_json(REST+quote(name.replace(" ","_"),safe="()"))
                    if o.get("type")=="disambiguation":
                        return {"requested":name,"category":cat,"failure":"disambiguation"}
                    if o.get("type") not in ("standard",None):
                        return {"requested":name,"category":cat,"failure":"nonstandard_article"}
                    title=o.get("title",name);txt=o.get("extract","")
                    url=o.get("content_urls",{}).get("desktop",{}).get("page")
                    rev=o.get("revision");api="rest"
                else:
                    query=urlencode({"action":"query","format":"json","redirects":1,
                       "prop":"extracts|info","exintro":1,"explaintext":1,
                       "inprop":"url","titles":name})
                    o=get_json(ACTION+"?"+query)
                    pages=o.get("query",{}).get("pages",[])
                    if isinstance(pages,dict):pages=list(pages.values())
                    page=pages[0]
                    if page.get("missing") or page.get("pageid",-1)<0:
                        return {"requested":name,"category":cat,"failure":"not_found"}
                    title=page.get("title",name);txt=page.get("extract","")
                    url=page.get("fullurl");rev=page.get("lastrevid");api="action"
                if not txt:raise ValueError("empty extract")
                if not url or not url.startswith(WIKI+"/wiki/"):
                    url=WIKI+"/wiki/"+quote(title.replace(" ","_"))
                return {"requested":name,"category":cat,"title":title,"extract":txt,
                        "url":url,"revision":rev,"fetch_api":api}
            except (HTTPError,URLError,TimeoutError,ValueError,KeyError,IndexError) as exc:
                errors.append(type(exc).__name__+":"+str(exc)[:70])
                if attempt==0:time.sleep(.25)
    return {"requested":name,"category":cat,"failure":"download_failed","errors":errors}

def clean_lead(article):
    if "failure" in article:return None,article["failure"]
    whole=re.sub(r"\s+"," ",html.unescape(article["extract"])).strip()
    if not whole:return None,"empty_text"
    if any(k in whole.casefold() for k in BAD):return None,"disambiguation_or_markup"
    if any(c in whole for c in "<>{}|") or "\ufffd" in whole:
        return None,"broken_markup"
    text=SPLIT_SENT.split(whole)[0].strip()
    if len(text)<56 or len(text.split())<8:return None,"too_short"
    if len(text)>355:return None,"too_long"
    if not text.endswith((".","!","?")):return None,"incomplete_first_sentence"
    if re.search(r"\[[0-9]+\]|https?://|\.\.\.",text):
        return None,"broken_citation"
    chars=re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]",text)
    if len(chars)/max(1,sum(not ch.isspace() for ch in text))<0.6:
        return None,"non_turkish_or_noise"
    if len(re.findall(r"\d+",text))>5:return None,"number_noise"
    title=article["title"].split(" (",1)[0]
    if len(title)>=4 and norm(title) not in norm(text[:max(75,3*len(title))]):
        return None,"title_mismatch"
    return text,"accepted"

def official_names():
    rows=[]
    for g,names in (("erkek",MEN),("kız",WOMEN)):
        for n in (1,2,3,5,8,10,15,20):
            if n>len(names):continue
            ans=g.capitalize()+" bebek isimleri: "+", ".join(names[:n])+"."
            for q in (str(n)+" "+g+" bebek ismi örneği ver.",
                      "Türkiye'de 2025 yılında "+str(n)+" "+g+" bebek ismi say."):
                rows.append({"prompt":q,"answer":ans,"topic":"2025 resmi isimler "+g,
                    "category":"resmi_isimler","source":"TÜİK ADNKS 2025",
                    "source_url":TUIK,"license":"public statistical facts"})
    q="20 tane erkek insan 8 tane kadın insan ismi yaz"
    a="Erkek isimleri: "+", ".join(MEN)+". Kız isimleri: "+", ".join(WOMEN[:8])+"."
    rows.append({"prompt":q,"answer":a,"topic":"2025 resmi isimler iki grup",
        "category":"resmi_isimler","source":"TÜİK ADNKS 2025",
        "source_url":TUIK,"license":"public statistical facts"})
    return rows

def make_qa(article,fact):
    title=article["title"].split(" (",1)[0]
    if article["category"]=="tarihi_kisiler":
        questions=[title+" kimdir?",title+" hakkında kısa bilgi verir misin?",
                   title+" kimdi?"]
    else:
        questions=[title+" nedir?",title+" hakkında kısa bilgi verir misin?",
                   title+" ne demektir?"]
    return [{"prompt":q,"answer":fact,"topic":article["requested"],
             "category":article["category"],"source":"Turkish Wikipedia",
             "source_url":article["url"],"source_revision":article.get("revision"),
             "license":"Wikipedia CC BY-SA / GFDL; per-row attribution",
             "source_sha256":digest(article["extract"])} for q in questions]

def write_jsonl(path,rows):
    path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows),
                    encoding="utf-8")

def prepare(seeds,raw,out,min_accepted):
    rejects=[];accepted=[];reason=Counter();categories=Counter()
    for article in raw:
        fact,why=clean_lead(article)
        if fact is None:
            rejects.append({"title":article["requested"],"category":article["category"],
                "reason":why,"url":article.get("url"),
                "preview":article.get("extract","")[:250]})
            reason[why]+=1
        else:
            accepted.append({**article,"fact":fact})
            categories[article["category"]]+=1
    if len(accepted)<min_accepted or len(categories)<6:
        raise RuntimeError("QUALITY GATE: accepted="+str(len(accepted))+
                           " rejects="+str(dict(reason))+" categories="+str(dict(categories)))
    must={"Köpek","Kedi","İnsan","Denklem"}
    actual={x["requested"] for x in accepted}
    if must-actual:raise RuntimeError("QUALITY GATE missing basics: "+repr(must-actual))
    splits={"train":[],"val":[],"test":[]};seen=set()
    topic_owner={}
    for a in accepted:
        h=int(digest("world-v1|"+norm(a["requested"]))[:8],16)%100
        split="test" if h<10 else "val" if h<20 else "train"
        topic_owner[a["requested"]]=split
        for qa in make_qa(a,a["fact"]):
            key=norm(qa["prompt"])
            if key in seen:continue
            seen.add(key);splits[split].append(qa)
    names=official_names()
    for qa in names:
        if norm(qa["prompt"]) in seen:raise RuntimeError("duplicate official prompt")
        seen.add(norm(qa["prompt"]))
        splits["train"].append(qa)
    if len(splits["train"])<200 or min(len(splits["val"]),len(splits["test"]))<12:
        raise RuntimeError("QUALITY GATE: too few split rows "+str({k:len(v) for k,v in splits.items()}))
    for split in splits:
        splits[split].sort(key=lambda x:(x["category"],x["topic"],x["prompt"]))
        write_jsonl(out/(split+".jsonl"),splits[split])
    write_jsonl(out/"downloaded_raw_pages.jsonl",raw)
    write_jsonl(out/"accepted_pages.jsonl",accepted)
    write_jsonl(out/"rejected_pages.jsonl",rejects)
    # Every Wikipedia page + paraphrase must stay in one split.
    for split,rows in splits.items():
        for r in rows:
            if r["source"]=="Turkish Wikipedia" and topic_owner[r["topic"]]!=split:
                raise RuntimeError("Topic leakage")
    rng=random.Random(470)
    examples=[]
    for category in sorted(categories):
        pool=[a for a in accepted if a["category"]==category]
        rng.shuffle(pool)
        examples.extend([{"category":category,"title":a["title"],
                          "fact":a["fact"],"url":a["url"]} for a in pool[:8]])
    audit={"dataset":"AMN grounded Turkish world-knowledge v1",
      "seed_topic_count":len(seeds),"download_success_count":sum("failure" not in a for a in raw),
      "accepted_page_count":len(accepted),"rejected_page_count":len(rejects),
      "rejection_reasons":dict(reason),"accepted_by_category":dict(categories),
      "training_topics":sum(s=="train" for s in topic_owner.values()),
      "split_qa_counts":{k:len(v) for k,v in splits.items()},
      "official_tuik_name_qa_count":len(names),
      "name_example_20_male_8_female":names[-1],
      "total_unique_prompts":len(seen),
      "source_sha256":hashlib.sha256((out/"downloaded_raw_pages.jsonl").read_bytes()).hexdigest(),
      "topic_disjoint":True,
      "audit_sample":examples,
      "rejected_examples":rejects[:80],
      "limit":"Heuristic screening cannot guarantee Wikipedia factual accuracy; no per-fact manual expert audit.",
      "wikipedia_attribution":"Wikipedia contributors, CC BY-SA / GFDL, source_url and revision on every QA"}
    (out/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest={"dataset":"Turkish Wikipedia curated titles + official TÜİK 2025 names",
      "source_file_sha256":audit["source_sha256"],
      "license":"Wikipedia CC BY-SA/GFDL and official statistical facts",
      "train":len(splits["train"]),"val":len(splits["val"]),"test":len(splits["test"]),
      "split_protocol":"topic-disjoint deterministic hash; official names train only",
      "quality_gate":"strict syntax source attribution and no unvetted Alpaca"}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Türkçe dünya bilgisi — veri kalite denetimi","",
        "Kaynak: Türkçe Wikipedia (355 maddelik ön-seçim) ve resmi TÜİK 2025 bebek isimleri.",
        "Tüm Wikipedia kayıtları URL ve varsa revision ile atıflıdır.",
        "Kaynak metin olduğu gibi, eğitimde sadece ilk eksiksiz tanım cümlesi kullanılır.",
        "Otomatik kalite denetimi yüzde 100 doğruluk iddiası taşımaz.","",
        "İndirilen: "+str(audit["download_success_count"])+"/"+str(len(seeds)),
        "Kabul: "+str(len(accepted)),"Ret: "+str(len(rejects)),
        "Train/val/test QA: "+"/".join(str(len(splits[k])) for k in ("train","val","test")),
        "İsim QA: "+str(len(names)),"","## Ret nedenleri"]
    lines +=["- "+k+": "+str(v) for k,v in reason.most_common()]
    lines+=["","## Kategori kapsamı"]
    lines +=["- "+k+": "+str(v) for k,v in sorted(categories.items())]
    lines+=["","## Kalite örnekleri"]
    for item in examples:
        lines +=["**"+item["category"]+" — "+item["title"]+"**",
                 item["fact"],item["url"],""]
    (out/"DATA_QUALITY_TR.md").write_text("\n".join(lines),encoding="utf-8")
    print("DATA_AUDIT_COMPLETE",json.dumps({k:v for k,v in audit.items()
        if k not in ("audit_sample","rejected_examples","name_example_20_male_8_female")},
        ensure_ascii=False),flush=True)
    return audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default=str(ROOT/"data/amn_world_seed_topics_v1.json"))
    p.add_argument("--output",default="data/amn_world_v1")
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--min-accepted",type=int,default=90)
    args=p.parse_args()
    seeds=json.loads(Path(args.seeds).read_text(encoding="utf-8"))["topics"]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    raw=[None]*len(seeds)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(fetch,s):i for i,s in enumerate(seeds)}
        progress=0
        for f in as_completed(jobs):
            raw[jobs[f]]=f.result();progress+=1
            if progress%50==0:print("DOWNLOAD",progress,len(seeds),flush=True)
    # Persist raw even if the quality gate fails for forensic analysis.
    write_jsonl(out/"downloaded_raw_pages.jsonl",raw)
    prepare(seeds,raw,out,args.min_accepted)
if __name__=="__main__":main()
