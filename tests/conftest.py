"""Shared pytest fixtures for EfficientQwen tests."""
from __future__ import annotations

import http.server
import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, ClassVar

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def free_port() -> int:
    return _free_port()


def load_serve_module():
    """Import scripts/serve.py as a fresh module."""
    spec = importlib.util.spec_from_file_location("serve_under_test", "scripts/serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeVLLMHandler(BaseHTTPRequestHandler):
    """Records POST bodies; returns OpenAI-shape canned responses."""

    received: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            data = json.loads(body)
        except Exception:
            data = {}
        type(self).received.append({"path": self.path, "body": data})

        if "chat/completions" in self.path:
            resp = {
                "object": "chat.completion",
                "model": "default",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            }
        else:
            resp = {
                "object": "text_completion",
                "model": "default",
                "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
            }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _Threaded(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture
def fake_vllm():
    """Spin up a fake vLLM on a random port. Yields (port, received_list)."""
    port = _free_port()
    _FakeVLLMHandler.received = []
    server = _Threaded(("127.0.0.1", port), _FakeVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield port, _FakeVLLMHandler.received
    server.shutdown()
    server.server_close()
