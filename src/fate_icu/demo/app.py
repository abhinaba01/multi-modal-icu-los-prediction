"""The demo page: enter a patient, read the prediction and the explanation.

    uv run python -m fate_icu.demo.app            # then open http://127.0.0.1:8765

Serves one self-contained page and two JSON routes on this computer only (127.0.0.1): nothing is
sent anywhere, there is no internet use, and the model is the one of ``fate_icu.demo.model`` (open
MIMIC-IV demo, a few dozen stays). The first start trains it from the local ``mimiciv_demo``
database (a second) and keeps it under the artifacts folder, so a later start needs no database.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fate_icu.demo.model import DemoModel, explain_patient, load, save, train_demo_model

PAGE = Path(__file__).with_name("page.html")
MAX_BODY = 20_000
CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'"
)
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def make_handler(demo: DemoModel) -> type[BaseHTTPRequestHandler]:
    options = {
        "fields": demo.options(),
        "trained_on": demo.n_train,
        "threshold_days": round(demo.threshold_days, 2),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "fate-icu-demo"

        def log_message(self, *args: Any) -> None:  # keep the terminal quiet; nothing here is worth logging
            return

        def _local_only(self) -> bool:
            host = (
                (self.headers.get("Host") or "").rsplit(":", 1)[0]
                if ":" in (self.headers.get("Host") or "")
                else ""
            )
            return host in LOCAL_HOSTS or (self.headers.get("Host") or "") in LOCAL_HOSTS

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if not self._local_only():
                return self._json(
                    HTTPStatus.FORBIDDEN, {"error": "this page is available on this computer only"}
                )
            if self.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/options":
                return self._json(HTTPStatus.OK, options)
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._local_only():
                return self._json(
                    HTTPStatus.FORBIDDEN, {"error": "this page is available on this computer only"}
                )
            if self.path != "/api/explain":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "send a small JSON object of values"})
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("send a JSON object of values")
                return self._json(HTTPStatus.OK, explain_patient(demo, body))
            except (ValueError, UnicodeDecodeError) as error:  # json.JSONDecodeError is a ValueError
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    return Handler


def serve(demo: DemoModel, port: int = 8765) -> ThreadingHTTPServer:
    """A server bound to this computer only (not started; call ``serve_forever``)."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(demo))


def get_model(retrain: bool = False) -> DemoModel:
    """The saved demo model, or one trained now from the local open-demo database."""
    from fate_icu.data.contract import load_contract
    from fate_icu.tabular.api import load_tables
    from fate_icu.utils.env import load_settings

    settings = load_settings()
    path = settings.artifacts_dir / "demo" / "demo_model.joblib"
    if path.exists() and not retrain:
        return load(path)
    if settings.machine != "laptop":
        raise SystemExit(
            "the demo model is trained on the open demo on the laptop only (FATE_MACHINE=laptop)"
        )
    if not settings.db_url:
        raise SystemExit("FATE_DB_URL is not set: load the open MIMIC-IV demo first (docs/laptop_setup.md)")
    contract = load_contract()
    tables = load_tables("postgres", contract=contract, settings=settings, cache=True, strict_sofa=True)
    demo = train_demo_model(tables, contract)
    save(demo, path)
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--retrain", action="store_true", help="train the demo model again")
    args = parser.parse_args(argv)
    demo = get_model(args.retrain)
    server = serve(demo, args.port)
    print(f"Demo model trained on {demo.n_train} open-demo stays.")
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
