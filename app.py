"""Web UI (B2 + B3): a React app (web/, prebuilt in web/dist) served by a standard-library HTTP server.

    python app.py            then open http://127.0.0.1:8000

Pages: "Customer chat" (B3 chatbot + agent panel) and "Classification" (one complaint with an
explanation of the decision, or a CSV upload with a dashboard). No Node is needed to run it: the build
is committed. To change the UI: cd web && npm install && npm run dev (proxies /api here) / npm run build.

Production backends: the fine-tuned model (or the TF-IDF baseline) with the LLM fallback when Ollama is
running. LLM clients are created with cache=False: no customer text is written to disk.
"""
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from orvana import api
from orvana.chat.manager import ChatBot
from orvana.chat.orders import DemoOrderService
from orvana.chat.sessions import SessionStore, TicketSink
from orvana.config import settings
from orvana.report.weekly import production_backends

BACKEND, LLM = production_backends(cache=False)  # live UI: no customer text in the LLM disk cache
ORDERS = DemoOrderService()
CHATBOT = ChatBot(BACKEND, LLM if settings.chat_use_llm else None, orders=ORDERS, sink=TicketSink())
SESSIONS = SessionStore()
DEMO_CUSTOMERS = {"", "DEMO-1", "DEMO-2", "DEMO-3"}  # stands in for the app login
DIST = (Path(__file__).resolve().parent / "web" / "dist").resolve()
SMALL_BODY, BATCH_BODY = 20_000, 5_000_000
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
NO_BUILD = (b"<!doctype html><meta charset=utf-8><title>Orvana</title><p>The UI is not built. Run "
            b"<code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>, then restart <code>python app.py</code>.")


class Handler(BaseHTTPRequestHandler):
    server_version = "Orvana"

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self, cap: int = SMALL_BODY) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > cap:
            raise api.BadRequest(f"request too large (limit {cap // 1000} KB)")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise api.BadRequest("expected a JSON object")
        return data

    def log_message(self, fmt, *args):  # never log request bodies (customer text); method + path only
        logging.info("%s %s", self.command, urlsplit(self.path).path)

    # ------------------------------------------------------------------ static UI
    def _static(self, path: str):
        if not (DIST / "index.html").exists():
            return self._send(503, NO_BUILD, "text/html; charset=utf-8")
        if path in ("/", "/index.html"):
            return self._send(200, (DIST / "index.html").read_bytes(), "text/html; charset=utf-8")
        target = (DIST / path.lstrip("/")).resolve()
        # only files inside web/dist (no ../ escapes), and only what the build produced
        if DIST not in target.parents or not target.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".woff2":
            ctype = "font/woff2"
        cache = "public, max-age=31536000, immutable" if "/assets/" in path else "no-cache"
        self._send(200, target.read_bytes(), ctype, cache)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                return self._json(200, api.health(BACKEND, LLM))
            if path == "/api/demo/customers":
                return self._json(200, api.demo_customers(ORDERS))
            if path.startswith("/api/"):
                return self._json(404, {"error": "not found"})
            self._static(path)
        except Exception as e:
            logging.exception("GET failed")
            self._json(500, {"error": f"failed: {type(e).__name__}"})

    # ------------------------------------------------------------------ API
    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/classify":
                return self._json(200, api.classify_one(str(self._body().get("text", "")), BACKEND, LLM))
            if path == "/api/classify/batch":
                body = self._body(BATCH_BODY)
                return self._json(200, api.classify_batch(body.get("csv"), BACKEND, sample=body.get("sample")))
            if path == "/api/chat":
                return self._chat(self._body())
            if path == "/api/chat/reset":
                SESSIONS.drop(str(self._body().get("session_id") or ""))
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})
        except (api.BadRequest, json.JSONDecodeError, UnicodeDecodeError) as e:
            self._json(400, {"error": str(e) if isinstance(e, api.BadRequest) else "invalid JSON"})
        except Exception as e:  # never leak a stack trace to the browser
            logging.exception("request failed")
            self._json(500, {"error": f"failed: {type(e).__name__}"})

    def _chat(self, body: dict):
        customer = str(body.get("customer_id") or "")
        customer = customer if customer in DEMO_CUSTOMERS else ""
        message = str(body.get("message") or "")[: settings.max_message_chars]
        state, lock = SESSIONS.get_or_create(str(body.get("session_id") or ""),
                                             lambda: CHATBOT.new_state(customer_id=customer or None))
        if (state.customer_id or "") != customer:  # a session never changes account
            SESSIONS.drop(state.session_id)
            state, lock = SESSIONS.get_or_create(None, lambda: CHATBOT.new_state(customer_id=customer or None))
        with lock:
            result = CHATBOT.turn(state, message)
        self._json(200, api.chat_payload(state, result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"classifier: {BACKEND.name}  |  LLM: {LLM.name if LLM else 'none (templates)'}  |  chat wording: "
          f"{settings.chat_phrasing}")
    if not (DIST / "index.html").exists():
        print("WARNING: web/dist is missing; build the UI: cd web && npm install && npm run build")
    print("open http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
