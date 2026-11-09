"""Local review app for the retrieval test set. Standard library only; localhost only.

Usage: cd 02-rag && uv run python review_app/server.py   then open http://127.0.0.1:8765
Verdicts are saved to testset/verdicts.json as you click.
"""
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.review import load_state, save_verdict

HOST, PORT = "127.0.0.1", 8765
QUESTIONS = ROOT / "testset" / "questions.json"
VERDICTS = ROOT / "testset" / "verdicts.json"
MANIFEST = ROOT / "corpus" / "manifest.json"
PAGE = Path(__file__).with_name("index.html")
MAX_BODY_BYTES = 20_000


def _titles() -> dict[str, str]:
    kept = json.loads(MANIFEST.read_text(encoding="utf-8"))["kept"]
    return {k["arxiv_id"]: k["title"] for k in kept}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            try:
                self._json(HTTPStatus.OK, {**load_state(QUESTIONS, VERDICTS), "titles": _titles()})
            except (ValueError, OSError) as exc:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/verdict":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_BYTES:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
        try:
            data = json.loads(self.rfile.read(length))
            verdicts = save_verdict(
                QUESTIONS, VERDICTS, data["id"], data["verdict"], data.get("note", "")
            )
        except (ValueError, KeyError, TypeError) as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        self._json(HTTPStatus.OK, {"verdicts": verdicts})

    def log_message(self, *_args) -> None:  # keep the terminal quiet
        pass


if __name__ == "__main__":
    print(f"Review app: http://{HOST}:{PORT}   (Ctrl+C to stop)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
