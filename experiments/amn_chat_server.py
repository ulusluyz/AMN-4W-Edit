"""Local AMN Turkish chat with frozen subword checkpoint and actual inference.

From repo root:
  python -m pip install -r requirements.txt
  python -m pip install sentencepiece==0.2.1
  python experiments/amn_chat_server.py --port 8090

Visit http://127.0.0.1:8090. No cloud/API calls. The model was trained on
mainly single-turn QA, so multi-turn context is experimental.
"""
from __future__ import annotations
import argparse
import json
import secrets
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from amn_open_turkish_sp import load_sp, restore

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "results/open_turkish_retrain_v2/best.pt"
DEFAULT_TOKENIZER = ROOT / "results/open_turkish_retrain_v2/tokenizer.model"

PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMN-4W Türkçe Sohbet</title>
<style>
:root{color-scheme:dark;font:16px system-ui,sans-serif}*{box-sizing:border-box}
body{margin:0;background:#0b121d;color:#e5eefb}
main{max-width:860px;height:100dvh;margin:auto;display:flex;flex-direction:column;border-inline:1px solid #263749}
header{padding:16px 20px;background:#141e2d;border-bottom:1px solid #263749}
h1{font-size:20px;margin:0 0 4px}small,.note{color:#a5b6ce;font-size:12px}
#messages{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px}
.bubble{padding:13px 16px;border-radius:15px;max-width:88%;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
.user{align-self:flex-end;background:#214e91}.bot{align-self:flex-start;background:#26354c}
.note{text-align:center}
footer{background:#141e2d;padding:12px 16px 20px;border-top:1px solid #263749}
.options{display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:12px;margin-bottom:10px}
input[type=number]{width:65px;border:1px solid #3a516e;background:#0f1b2a;color:#fff;border-radius:7px;padding:5px}
.compose{display:flex;gap:8px}
textarea{flex:1;resize:vertical;min-height:58px;max-height:150px;background:#0f1b2a;border:1px solid #3a516e;border-radius:10px;color:#fff;padding:11px;font:inherit}
button{background:#4581d0;border:0;border-radius:9px;padding:10px 18px;color:#fff;font:inherit;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}#status{margin-top:8px;font-size:12px;color:#a5b6ce}
</style></head><body><main>
<header><h1>AMN-4W — Yerel Türkçe Sohbet</h1><small>
Deneysel yerel AMN · Yanıtlar doğrudan eğitilmiş modelden gelir · İnternet API'si yok
</small></header>
<section id="messages" aria-live="polite">
<div class="note">Bu model henüz güvenilir bir Türkçe asistan değildir. Tekrarlı ve anlamsız cevaplar üretebilir.</div>
</section>
<footer><div class="options">
<label><input type="checkbox" id="history"> Önceki iki turu dahil et (deneysel)</label>
<label>Token <input type="number" id="tokens" min="16" max="160" value="96"></label>
<label>Sıcaklık <input type="number" id="temp" min="0" max="1.4" step="0.1" value="0.7"></label>
</div><div class="compose">
<textarea id="prompt" maxlength="2000" placeholder="Modeline Türkçe bir soru sor..."></textarea>
<button id="send">Gönder</button>
</div><div id="status"></div></footer></main>
<script>
const box=document.getElementById("messages"),input=document.getElementById("prompt");
const btn=document.getElementById("send"),status=document.getElementById("status"),turns=[];
function bubble(text,kind){
 const div=document.createElement("div");div.className="bubble "+kind;
 div.textContent=text;box.appendChild(div);box.scrollTop=box.scrollHeight;
}
async function send(){
 const q=input.value.trim();if(!q||btn.disabled)return;
 const prior=document.getElementById("history").checked?turns.slice(-2):[];
 bubble(q,"user");input.value="";btn.disabled=true;status.textContent="Model yanıt oluşturuyor...";
 try{
  const response=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({message:q,history:prior,max_new_tokens:Number(document.getElementById("tokens").value),
   temperature:Number(document.getElementById("temp").value)})});
  const out=await response.json();
  if(!response.ok)throw Error(out.error||"Sunucu hatası");
  bubble(out.answer||"(Model boş yanıt üretti)","bot");
  turns.push({user:q,assistant:out.answer||""});
  status.textContent=out.tokens_generated+" token · "+out.duration_s+" sn · "+(out.eos?"EOS":"uzunluk sınırı");
 }catch(e){bubble("Hata: "+e.message,"bot");status.textContent="İstek başarısız";}
 finally{btn.disabled=false;input.focus();}
}
btn.addEventListener("click",send);
input.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}});
</script></body></html>"""


def prepare_prompt(message, history, tokenizer, max_tokens=190):
    """Truncate old conversation first, respecting training context length."""
    if not isinstance(message,str):
        raise ValueError("Mesaj metin olmalı")
    message=message.strip()
    if not message or len(message)>2000:
        raise ValueError("Mesaj 1 ile 2000 karakter arasında olmalı")
    if history is not None and not isinstance(history,list):
        raise ValueError("Geçmiş mesajlar liste olmalı")
    pairs=[]
    for pair in (history or [])[-2:]:
        if not isinstance(pair,dict):
            continue
        user=pair.get("user")
        assistant=pair.get("assistant")
        if isinstance(user,str) and isinstance(assistant,str) and user.strip() and assistant.strip():
            pairs.append((user[:600].strip(),assistant[:600].strip()))
    while True:
        prefix="".join(user+"\nAsistan: "+assistant+"\nKullanıcı: " for user,assistant in pairs)
        content=prefix+message
        length=len(tokenizer.encode("Kullanıcı: "+content+"\nAsistan: "))
        if length<=max_tokens:
            return content
        if pairs:pairs.pop(0)
        else:raise ValueError("Soru bu modelin bağlam sınırını aşıyor")


class Engine:
    def __init__(self,model_file,tokenizer_file,threads=2):
        torch.set_num_threads(threads)
        self.sp=load_sp(tokenizer_file)
        self.model,self.metadata=restore(model_file)
        self.model.eval()
        self._lock=threading.Lock()

    def reply(self,message,history=None,max_new_tokens=96,temperature=0.7):
        prompt=prepare_prompt(message,history,self.sp)
        count=max(16,min(160,int(max_new_tokens)))
        temp=max(0,min(1.4,float(temperature)))
        started=time.perf_counter()
        with self._lock:
            result=self.model.generate(self.sp,prompt,max_new_tokens=count,
                temperature=temp,top_k=20,seed=secrets.randbelow(2**31))
        return {"answer":result["text"],"tokens_generated":result["tokens_generated"],
                "eos":result["eos"],"duration_s":round(time.perf_counter()-started,3)}


def handler_class(engine):
    class Handler(BaseHTTPRequestHandler):
        def deliver(self,status,obj,kind="application/json; charset=utf-8"):
            raw=(obj.encode("utf-8") if isinstance(obj,str)
                 else json.dumps(obj,ensure_ascii=False).encode("utf-8"))
            self.send_response(status)
            self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(raw)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            if self.path=="/":self.deliver(200,PAGE,"text/html; charset=utf-8")
            elif self.path=="/api/health":
                self.deliver(200,{"model":"AMNSubwordLM","status":"ready",
                                  "best_step":engine.metadata.get("best_step")})
            else:self.deliver(404,{"error":"Not found"})
        def do_POST(self):
            if self.path!="/api/chat":return self.deliver(404,{"error":"Not found"})
            try:
                n=int(self.headers.get("Content-Length","0"))
                if n<1 or n>65536:
                    return self.deliver(413,{"error":"Request too large"})
                data=json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(data,dict):
                    raise ValueError("JSON nesnesi bekleniyor")
                self.deliver(200,engine.reply(data.get("message",""),data.get("history"),
                    data.get("max_new_tokens",96),data.get("temperature",0.7)))
            except (ValueError,TypeError,UnicodeError) as e:
                self.deliver(400,{"error":str(e)})
            except Exception:
                self.deliver(500,{"error":"Model çıkarımında beklenmeyen hata"})
    return Handler


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default=str(DEFAULT_MODEL))
    ap.add_argument("--tokenizer",default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--port",type=int,default=8090)
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--threads",type=int,default=2)
    args=ap.parse_args()
    if not Path(args.checkpoint).is_file() or not Path(args.tokenizer).is_file():
        ap.error("Checkpoint/tokenizer bulunamadı; depoyu güncelleyin")
    engine=Engine(args.checkpoint,args.tokenizer,args.threads)
    server=ThreadingHTTPServer((args.host,args.port),handler_class(engine))
    print("AMN Türkçe sohbet: http://"+args.host+":"+str(args.port),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=="__main__":main()
