"""Smoke/regression checks for the AMN local chat, using actual frozen v2 weights.

Execute after installing torch and sentencepiece:
    python -m pytest -q tests/test_amn_chat_server.py
"""
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
pytest.importorskip("sentencepiece")

from amn_chat_server import (
    PAGE, Engine, DEFAULT_MODEL, DEFAULT_TOKENIZER, handler_class, prepare_prompt
)


class FakeTokenizer:
    def encode(self, text):
        return list(text)


def test_chat_page_no_external_api_and_has_send_controls():
    assert 'id="send"' in PAGE
    assert 'id="prompt"' in PAGE
    assert 'fetch("/api/chat"' in PAGE
    assert "https://" not in PAGE
    assert "textContent=text" in PAGE


def test_prompt_history_and_bounds():
    t = FakeTokenizer()
    assert prepare_prompt("Merhaba", None, t, 190) == "Merhaba"
    past = [{"user":"Önceki soru","assistant":"Önceki cevap"}]
    result = prepare_prompt("Yeni soru", past, t, 190)
    assert "Önceki cevap" in result
    assert result.endswith("Yeni soru")
    with pytest.raises(ValueError):
        prepare_prompt("", None, t)
    with pytest.raises(ValueError):
        prepare_prompt("x"*3000, None, t)


def test_actual_checkpoint_chat_http_roundtrip():
    assert DEFAULT_MODEL.is_file()
    assert DEFAULT_TOKENIZER.is_file()
    engine = Engine(DEFAULT_MODEL, DEFAULT_TOKENIZER, threads=1)
    server = ThreadingHTTPServer(("127.0.0.1",0), handler_class(engine))
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=25) as resp:
            h=json.load(resp)
        assert h["status"]=="ready"
        request=Request(
            f"http://127.0.0.1:{port}/api/chat",
            data=json.dumps({"message":"Merhaba, nasılsın?",
                             "max_new_tokens":16,"temperature":0}).encode(),
            headers={"Content-Type":"application/json"},method="POST",
        )
        with urlopen(request, timeout=60) as resp:
            output=json.load(resp)
        assert isinstance(output["answer"],str)
        assert 0<=output["tokens_generated"]<=16
        assert isinstance(output["eos"],bool)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
