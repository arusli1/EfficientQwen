"""Tests for scripts/serve.py — flag construction, routing, end-to-end via fake vLLM.

These catch the bugs that would otherwise burn a daily submission:
- typo'd vLLM CLI flag names
- wrong /invocations routing (chat vs completion)
- broken /ping → 200 readiness gating
- response-shape mismatches with the AdaptFM eval contract

What these do NOT catch: vLLM 0.19 actually accepting our flag combo, model
loading, MTP acceptance rate, quality gates. A10G validates those.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from conftest import load_serve_module


# ─── Unit: build_vllm_cmd ─────────────────────────────────────────────────────
def _clear_vllm_env(monkeypatch):
    for k in list(os.environ.keys()):
        if k.startswith("VLLM_"):
            monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("env,expected,absent", [
    # Phase-0-equivalent: nothing set → FP16, block_size=16, no --quantization
    # (vLLM auto-detects compressed-tensors/AWQ from config.json; forcing
    # awq_marlin on cyankiwi's compressed-tensors weights would crash the load).
    (
        {},
        ["--model", "--max-num-seqs", "--max-model-len", "--gpu-memory-utilization",
         "--block-size", "16"],
        ["--quantization", "--language-model-only", "--kv-cache-dtype",
         "--speculative-config", "--enable-chunked-prefill", "--enable-prefix-caching",
         "--enforce-eager", "--num-gpu-blocks-override"],
    ),
    # Phase 1 — skip-vision win
    (
        {"VLLM_LANGUAGE_MODEL_ONLY": "1"},
        ["--language-model-only"],
        ["--kv-cache-dtype"],
    ),
    # Phase 1 — chunked prefill + prefix caching
    (
        {"VLLM_ENABLE_CHUNKED_PREFILL": "1", "VLLM_ENABLE_PREFIX_CACHING": "1"},
        ["--enable-chunked-prefill", "--enable-prefix-caching"],
        [],
    ),
    # Phase 2 — KV cache fp8
    (
        {"VLLM_KV_CACHE_DTYPE": "fp8"},
        ["--kv-cache-dtype", "fp8"],
        ["--speculative-config"],
    ),
    # Phase 3 — MTP speculative decoding (method is "mtp", not "qwen3_5_mtp")
    # CLI flag is --cudagraph-capture-sizes (NOT --cuda-graph-sizes which
    # doesn't exist in vLLM 0.19 — verified against source). nargs="+", so
    # each size is a separate argv entry (NOT a single CSV string).
    (
        {
            "VLLM_SPECULATIVE_CONFIG":
                '{"method":"mtp","num_speculative_tokens":2}',
            "VLLM_CUDAGRAPH_CAPTURE_SIZES": "3,6,9,12,24",
        },
        ["--speculative-config", "--cudagraph-capture-sizes", "3", "6", "9", "12", "24"],
        ["--cuda-graph-sizes"],  # absence guard against regression
    ),
    # Production-tuning: skip startup memory profile
    (
        {"VLLM_NUM_GPU_BLOCKS_OVERRIDE": "2400"},
        ["--num-gpu-blocks-override", "2400"],
        [],
    ),
    # Compilation config (alternative way to control graph capture)
    (
        {"VLLM_COMPILATION_CONFIG":
             '{"mode":3,"cudagraph_capture_sizes":[3,6,9,12,24]}'},
        ["--compilation-config"],
        [],
    ),
    # 2026-05-23 — performance mode (latency-optimized for the baseline workload)
    (
        {"VLLM_PERFORMANCE_MODE": "interactivity"},
        ["--performance-mode", "interactivity"],
        [],
    ),
    # 2026-05-23 — optimization level (O3 enables fuse_attn_quant for INT4)
    (
        {"VLLM_OPTIMIZATION_LEVEL": "O3"},
        ["--optimization-level", "O3"],
        [],
    ),
    # 2026-05-23 — mamba_cache_mode (hybrid-model knob; "none" can reduce
    # per-page overhead at single-stream for cyankiwi's mamba2+attention mix)
    (
        {"VLLM_MAMBA_CACHE_MODE": "none"},
        ["--mamba-cache-mode", "none"],
        [],
    ),
    # 2026-05-23 — KernelConfig JSON dict (enable_flashinfer_autotune etc.)
    (
        {"VLLM_KERNEL_CONFIG": '{"enable_flashinfer_autotune":true}'},
        ["--kernel-config", '{"enable_flashinfer_autotune":true}'],
        [],
    ),
    # 2026-05-23 — AttentionConfig JSON dict (TREE_ATTN backend, prefill_decode split, etc.)
    (
        {"VLLM_ATTENTION_CONFIG": '{"backend":"TREE_ATTN"}'},
        ["--attention-config", '{"backend":"TREE_ATTN"}'],
        [],
    ),
])
def test_build_vllm_cmd_flag_combinations(monkeypatch, env, expected, absent):
    _clear_vllm_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    serve = load_serve_module()
    cmd = serve.build_vllm_cmd()
    for flag in expected:
        assert flag in cmd, f"missing {flag!r} in {cmd}"
    for flag in absent:
        assert flag not in cmd, f"unexpected {flag!r} in {cmd}"


def test_mtp_speculative_config_uses_canonical_method_name(monkeypatch):
    """The canonical MTP method for Qwen3.5 is 'mtp' — vLLM docs confirm this."""
    _clear_vllm_env(monkeypatch)
    cfg = '{"method":"mtp","num_speculative_tokens":2}'
    monkeypatch.setenv("VLLM_SPECULATIVE_CONFIG", cfg)
    serve = load_serve_module()
    cmd = serve.build_vllm_cmd()
    idx = cmd.index("--speculative-config")
    parsed = json.loads(cmd[idx + 1])
    assert parsed["method"] == "mtp"
    assert parsed["num_speculative_tokens"] == 2


def test_default_quantization_omitted(monkeypatch):
    """vLLM auto-detects quantization from config.json by default.

    cyankiwi is compressed-tensors, NOT AWQ — forcing --quantization awq_marlin
    on its weight format crashes the load (different shape/scale layout). When
    VLLM_QUANTIZATION is unset, omit the flag so vLLM's auto path dispatches
    compressed-tensors → MarlinLinearKernel (same speed as awq_marlin).
    """
    _clear_vllm_env(monkeypatch)
    serve = load_serve_module()
    cmd = serve.build_vllm_cmd()
    assert "--quantization" not in cmd


def test_explicit_quantization_passes_through(monkeypatch):
    """Setting VLLM_QUANTIZATION explicitly adds the flag — for the rare case
    where a weight variant truly needs an override (e.g., a real AWQ checkpoint)."""
    _clear_vllm_env(monkeypatch)
    monkeypatch.setenv("VLLM_QUANTIZATION", "awq_marlin")
    serve = load_serve_module()
    cmd = serve.build_vllm_cmd()
    idx = cmd.index("--quantization")
    assert cmd[idx + 1] == "awq_marlin"


def test_cuda_graph_sizes_divisible_by_three_for_mtp_2(monkeypatch):
    """vLLM bug #28015: graph sizes must be multiples of (1+num_spec_tokens)."""
    sizes = [3, 6, 9, 12, 24]
    for s in sizes:
        assert s % 3 == 0, f"{s} would be silently filtered by vLLM"


def test_enforce_eager_off_by_default(monkeypatch):
    """We want CUDA graphs ON by default — flag should be absent."""
    _clear_vllm_env(monkeypatch)
    serve = load_serve_module()
    cmd = serve.build_vllm_cmd()
    assert "--enforce-eager" not in cmd


def test_paroquant_bootstrap_swap(monkeypatch):
    """VLLM_USE_PAROQUANT=1 → swap `python -m vllm…` to `python -c bootstrap …args`.

    The PARO quantization config plugin must be `register()`-ed BEFORE vllm
    parses config.json. We can't pip-install paroquant into the sealed env,
    so we vendor it on PYTHONPATH and inject a tiny `-c` bootstrap that
    imports + calls register, then runpys vllm's api_server entrypoint with
    alter_sys=True so argparse picks up the original CLI args.
    """
    _clear_vllm_env(monkeypatch)
    monkeypatch.setenv("VLLM_USE_PAROQUANT", "1")
    monkeypatch.setenv("VLLM_MODEL", "/tmp/whatever")
    serve = load_serve_module()

    captured: dict = {}

    class _FakePopen:
        def __init__(self, cmd):
            captured["cmd"] = cmd

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(serve.subprocess, "Popen", _FakePopen)
    serve.start_vllm()

    cmd = captured["cmd"]
    # First element is the python interpreter, then `-c`, then the bootstrap.
    assert cmd[1] == "-c", f"expected -c bootstrap, got {cmd[:3]!r}"
    assert "paroquant.inference.backends.vllm" in cmd[2]
    assert "register()" in cmd[2]
    assert "vllm.entrypoints.openai.api_server" in cmd[2]
    # The vllm CLI args must still be passed through after the bootstrap.
    assert "--model" in cmd
    # The original -m form must be gone.
    assert "-m" not in cmd[:3]


def test_no_paroquant_bootstrap_when_flag_unset(monkeypatch):
    """Default path: `python -m vllm.entrypoints.openai.api_server …` (no -c)."""
    _clear_vllm_env(monkeypatch)
    monkeypatch.setenv("VLLM_MODEL", "/tmp/whatever")
    serve = load_serve_module()

    captured: dict = {}

    class _FakePopen:
        def __init__(self, cmd):
            captured["cmd"] = cmd

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(serve.subprocess, "Popen", _FakePopen)
    serve.start_vllm()

    cmd = captured["cmd"]
    assert cmd[1] == "-m"
    assert cmd[2] == "vllm.entrypoints.openai.api_server"
    assert "-c" not in cmd[:3]


# ─── Unit: route_request ──────────────────────────────────────────────────────
def test_route_completion_via_v1_completions():
    serve = load_serve_module()
    data = {"prompt": "hello", "max_tokens": 64, "temperature": 0.0}
    path, payload = serve.route_request(data, "/v1/completions")
    assert path == "/v1/completions"
    assert payload["prompt"] == "hello"
    assert payload["max_tokens"] == 64


def test_route_chat_via_messages_on_invocations():
    serve = load_serve_module()
    msgs = [{"role": "user", "content": "hi"}]
    path, payload = serve.route_request(
        {"messages": msgs, "max_tokens": 32}, "/invocations",
    )
    assert path == "/v1/chat/completions"
    assert payload["messages"] == msgs
    assert payload["max_tokens"] == 32
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_route_thinking_true_sets_template_kwarg_and_bumps_max_tokens(monkeypatch):
    """thinking=true → enable_thinking flag through, max_tokens default 12288 for GPQA.

    Requires VLLM_MAX_MODEL_LEN large enough to fit 12288 + 1024 prompt budget,
    else the defensive cap (see _cap_max_tokens) clamps the request.
    """
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "16384")
    serve = load_serve_module()
    path, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "hi"}], "thinking": True},
        "/invocations",
    )
    assert path == "/v1/chat/completions"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["max_tokens"] == 12288


def test_route_thinking_max_tokens_capped_when_model_len_too_small(monkeypatch):
    """Regression: an earlier baseline.env had max_model_len=8448; the thinking-mode
    default max_tokens=12288 overflows that and vLLM returned 400 Bad Request → GPQA
    score=0.0. The cap brings it down to (8448 - 1024 prompt budget) = 7424."""
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8448")
    serve = load_serve_module()
    _, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "hi"}], "thinking": True},
        "/invocations",
    )
    assert payload["max_tokens"] == 7424


def test_route_caller_max_tokens_capped_too(monkeypatch):
    """Caller-supplied max_tokens is also capped, not only the default."""
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8448")
    serve = load_serve_module()
    _, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10_000},
        "/invocations",
    )
    assert payload["max_tokens"] == 7424


def test_route_max_tokens_unchanged_when_safe(monkeypatch):
    """Small max_tokens passes through unchanged."""
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8448")
    serve = load_serve_module()
    _, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 256},
        "/invocations",
    )
    assert payload["max_tokens"] == 256


def test_route_completion_max_tokens_also_capped(monkeypatch):
    """The /v1/completions path also caps to defend against the same overflow."""
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "8448")
    serve = load_serve_module()
    _, payload = serve.route_request(
        {"prompt": "x", "max_tokens": 20_000},
        "/v1/completions",
    )
    assert payload["max_tokens"] == 7424


def test_route_v1_completions_forces_completion_even_with_messages():
    """/v1/completions hard-forces completion shape regardless of body."""
    serve = load_serve_module()
    path, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "hi"}], "prompt": "fallback"},
        "/v1/completions",
    )
    assert path == "/v1/completions"
    assert "messages" not in payload
    assert payload["prompt"] == "fallback"


def test_route_list_prompt_flattens_to_first_element():
    """lm-eval sometimes sends prompt=['x']; we unwrap."""
    serve = load_serve_module()
    _, payload = serve.route_request({"prompt": ["only"]}, "/v1/completions")
    assert payload["prompt"] == "only"


def test_route_thinking_via_chat_template_kwargs_alias():
    """Some clients pass chat_template_kwargs.enable_thinking directly."""
    serve = load_serve_module()
    _, payload = serve.route_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": True},
        },
        "/invocations",
    )
    assert payload["chat_template_kwargs"]["enable_thinking"] is True


# ─── Integration: HTTP handler against fake vLLM ──────────────────────────────
def _start_handler(monkeypatch, ext_port: int, fake_vllm_port: int):
    monkeypatch.setenv("VLLM_PORT", str(ext_port))
    monkeypatch.setenv("VLLM_INTERNAL_PORT", str(fake_vllm_port))
    serve = load_serve_module()
    serve._vllm_ready = True
    server = serve._ThreadingHTTPServer(("127.0.0.1", ext_port), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    return server, serve


def _http_post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def _http_get(url: str) -> int:
    try:
        return urllib.request.urlopen(url, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


def _http_get_body(url: str) -> tuple[int, bytes, str]:
    """Return (status, body, content_type) for a GET."""
    try:
        r = urllib.request.urlopen(url, timeout=5)
        return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


@pytest.mark.timeout(10)
def test_metrics_proxies_to_internal_vllm(monkeypatch, fake_vllm, free_port):
    """Handler.do_GET("/metrics") must proxy to internal vLLM /metrics.

    scripts/profile_model.py + scripts/bench_latency.py query
    `<external>/metrics` for Prometheus per-phase counters + MTP-acceptance
    deltas. If the proxy is missing, those scripts silently fall back to
    their (broken) coarse model.
    """
    fake_port, _ = fake_vllm
    # _start_handler creates a FRESH serve module instance — monkeypatch
    # AFTER it to patch the right module's _proxy_get.
    server, serve = _start_handler(monkeypatch, free_port, fake_port)
    try:
        def _fake_proxy_get(path):
            if path == "/metrics":
                return (
                    b"# HELP vllm:e2e_request_latency_seconds vllm e2e latency\n"
                    b"# TYPE vllm:e2e_request_latency_seconds histogram\n"
                    b"vllm:e2e_request_latency_seconds_sum 12.34\n"
                )
            raise RuntimeError(f"unexpected proxy path: {path}")

        monkeypatch.setattr(serve, "_proxy_get", _fake_proxy_get)
        status, body, ctype = _http_get_body(f"http://127.0.0.1:{free_port}/metrics")
        assert status == 200, f"expected 200, got {status}"
        assert "text/plain" in ctype, f"Prometheus format expected, got {ctype}"
        assert b"vllm:e2e_request_latency_seconds_sum" in body
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_metrics_returns_503_when_vllm_proxy_fails(monkeypatch, fake_vllm, free_port):
    """If /metrics proxy raises (e.g., vLLM cold-starting), return 503 — not
    500. Callers (profile_model.py, bench_latency.py) handle 503 gracefully."""
    fake_port, _ = fake_vllm
    server, serve = _start_handler(monkeypatch, free_port, fake_port)
    try:
        def _failing_proxy_get(path):
            raise ConnectionError("vLLM not up yet")

        monkeypatch.setattr(serve, "_proxy_get", _failing_proxy_get)
        status = _http_get(f"http://127.0.0.1:{free_port}/metrics")
        assert status == 503, f"expected 503 on backend failure, got {status}"
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_invocations_chat_proxies_to_vllm_chat(monkeypatch, fake_vllm, free_port):
    fake_port, received = fake_vllm
    server, _ = _start_handler(monkeypatch, free_port, fake_port)
    try:
        resp = _http_post(
            f"http://127.0.0.1:{free_port}/invocations",
            {"messages": [{"role": "user", "content": "test"}], "max_tokens": 16},
        )
        assert "choices" in resp
        assert resp["choices"][0]["message"]["content"] == "ok"
        assert len(received) == 1
        assert received[0]["path"] == "/v1/chat/completions"
        assert received[0]["body"]["messages"][0]["content"] == "test"
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_v1_completions_proxies_to_vllm_completions(monkeypatch, fake_vllm, free_port):
    fake_port, received = fake_vllm
    server, _ = _start_handler(monkeypatch, free_port, fake_port)
    try:
        resp = _http_post(
            f"http://127.0.0.1:{free_port}/v1/completions",
            {"prompt": "hello", "max_tokens": 4},
        )
        assert "choices" in resp
        assert resp["choices"][0]["text"] == "ok"
        assert len(received) == 1
        assert received[0]["path"] == "/v1/completions"
        assert received[0]["body"]["prompt"] == "hello"
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_ping_503_when_not_ready(monkeypatch, free_port):
    monkeypatch.setenv("VLLM_PORT", str(free_port))
    serve = load_serve_module()
    serve._vllm_ready = False
    server = serve._ThreadingHTTPServer(("127.0.0.1", free_port), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        assert _http_get(f"http://127.0.0.1:{free_port}/ping") == 503
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_ping_200_when_ready(monkeypatch, free_port):
    monkeypatch.setenv("VLLM_PORT", str(free_port))
    serve = load_serve_module()
    serve._vllm_ready = True
    server = serve._ThreadingHTTPServer(("127.0.0.1", free_port), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        assert _http_get(f"http://127.0.0.1:{free_port}/ping") == 200
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_thinking_payload_propagates_to_vllm(monkeypatch, fake_vllm, free_port):
    """Ensures GPQA-style thinking=true requests carry through to vLLM."""
    fake_port, received = fake_vllm
    server, _ = _start_handler(monkeypatch, free_port, fake_port)
    try:
        _http_post(
            f"http://127.0.0.1:{free_port}/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Reason."}],
                "thinking": True,
                "max_tokens": 200,
            },
        )
        assert received[0]["body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert received[0]["body"]["max_tokens"] == 200
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_v1_models_proxies_to_vllm(monkeypatch, free_port):
    """lm-eval probes /v1/models on init — we must proxy it."""
    import http.server
    import socket
    from socketserver import ThreadingMixIn

    # Spin a fake vLLM that responds to /v1/models GET
    s = socket.socket()
    s.bind(("", 0))
    fake_port = s.getsockname()[1]
    s.close()

    class _M(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/v1/models":
                body = json.dumps({"object": "list",
                                   "data": [{"id": "default", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

    class _T(ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    backend = _T(("127.0.0.1", fake_port), _M)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    time.sleep(0.05)
    server, _ = _start_handler(monkeypatch, free_port, fake_port)
    try:
        resp = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{free_port}/v1/models", timeout=5).read())
        assert resp["object"] == "list"
        assert any(m["id"] == "default" for m in resp["data"])
    finally:
        server.shutdown()
        backend.shutdown()


@pytest.mark.timeout(10)
def test_v1_models_falls_back_to_canned_when_backend_unreachable(monkeypatch, free_port):
    """If proxying /v1/models fails, we still return a sane canned response."""
    monkeypatch.setenv("VLLM_PORT", str(free_port))
    monkeypatch.setenv("VLLM_INTERNAL_PORT", "1")  # almost certainly unused
    serve = load_serve_module()
    serve._vllm_ready = True
    server = serve._ThreadingHTTPServer(("127.0.0.1", free_port), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.05)
    try:
        resp = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{free_port}/v1/models", timeout=5).read())
        assert resp["object"] == "list"
        assert resp["data"][0]["id"] == "default"
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_do_post_returns_404_for_unknown_path(monkeypatch, free_port):
    """Defence-in-depth: misrouted POSTs must not 500 with a stack trace."""
    monkeypatch.setenv("VLLM_PORT", str(free_port))
    serve = load_serve_module()
    serve._vllm_ready = True
    server = serve._ThreadingHTTPServer(("127.0.0.1", free_port), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.05)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{free_port}/nope",
            data=b'{"x":1}', headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()


@pytest.mark.timeout(10)
def test_do_post_returns_500_when_backend_unreachable(monkeypatch, free_port):
    """If the upstream vLLM is unreachable, surface a 500 with a JSON body."""
    monkeypatch.setenv("VLLM_PORT", str(free_port))
    monkeypatch.setenv("VLLM_INTERNAL_PORT", "1")  # nobody listening here
    serve = load_serve_module()
    serve._vllm_ready = True
    server = serve._ThreadingHTTPServer(("127.0.0.1", free_port), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.05)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{free_port}/invocations",
            data=json.dumps({"prompt": "x", "max_tokens": 1}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected 500")
        except urllib.error.HTTPError as e:
            assert e.code == 500
            body = json.loads(e.read())
            assert "error" in body
    finally:
        server.shutdown()


def test_wait_for_vllm_health_times_out_quickly(monkeypatch):
    """If the internal vLLM never comes up, the health probe must give up."""
    serve = load_serve_module()
    # Internal port pointing at nothing — wait_for_vllm_health should return False
    # well before its 590s default. We pass timeout=1 to keep the test fast.
    monkeypatch.setenv("VLLM_INTERNAL_PORT", "1")
    # Reload constants after env override:
    serve = load_serve_module()
    t0 = time.perf_counter()
    ok = serve.wait_for_vllm_health(timeout=1)
    elapsed = time.perf_counter() - t0
    assert ok is False
    assert elapsed < 3, f"health probe took too long: {elapsed:.2f}s"


def test_prewarm_returns_false_when_backend_dead(monkeypatch):
    """If prewarm's first request errors out, return False so /ping stays 503."""
    serve = load_serve_module()
    monkeypatch.setenv("VLLM_INTERNAL_PORT", "1")  # closed
    serve = load_serve_module()
    assert serve.prewarm() is False


@pytest.mark.timeout(10)
def test_ping_flips_to_503_when_vllm_subprocess_dies(monkeypatch, free_port):
    """Critical robustness: if vLLM crashes after ready, /ping must reflect it."""
    import subprocess
    monkeypatch.setenv("VLLM_PORT", str(free_port))
    serve = load_serve_module()
    serve._vllm_ready = True
    # Simulate a dead subprocess (returncode set)
    dummy = subprocess.Popen(["true"])
    dummy.wait()
    serve._vllm_proc = dummy
    server = serve._ThreadingHTTPServer(("127.0.0.1", free_port), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.05)
    try:
        # Even though _vllm_ready=True, the dead subprocess should flip /ping → 503
        assert _http_get(f"http://127.0.0.1:{free_port}/ping") == 503
    finally:
        server.shutdown()


# ─── Unit: VLLM_VOCAB_REMAP_SIDECAR env switch + remap behaviour ──────────────
# When VLLM_VOCAB_REMAP_SIDECAR points at a readable JSON sidecar (built by
# scripts/prune_vocab_v2.py), the
# router tokenizes locally with the original (full-vocab) HF tokenizer, remaps
# each ID via orig_to_new, sends prompt_token_ids to /v1/completions (with
# logprobs=1 + return_tokens_as_token_ids=true for response remap), then maps
# the per-choice token_ids back via new_to_orig and decodes locally. This
# sidesteps vLLM tokenizing against the (wrong, full-vocab) tokenizer that
# ships with v64k weights.
class _FakeTokenizer:
    """Minimal HF-tokenizer stand-in: identity encode (one char → one ID),
    decode that just joins via | with the orig IDs. Avoids loading transformers
    in unit tests. Used via monkeypatch to substitute the AutoTokenizer load.
    """
    def __init__(self, eos_token_id: int | None = 7):
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Map characters → orig IDs 100, 200, 300, … so tests can assert a clear
        # mapping. Spaces and other characters get an "other" bucket at 999.
        out: list[int] = []
        for ch in text:
            if ch in "abcdefghijklmnopqrstuvwxyz":
                out.append(100 * (ord(ch) - ord("a") + 1))
            else:
                out.append(999)
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        # Join orig IDs with `|` so tests can spot whether the decode used the
        # local tokenizer (orig IDs) or vLLM's wrong tokenizer (text=...).
        return "|".join(f"o{i}" for i in ids)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            chat_template_kwargs=None) -> str:
        # Trivial template: concatenate contents with a separator. The test
        # only cares that this returns *something* and that the returned text
        # gets tokenized.
        parts = []
        for m in messages:
            parts.append(str(m.get("content", "")))
        return " ".join(parts)


def _write_sidecar(tmp_path, orig_to_new=None, new_to_orig=None,
                   missing_keys=False):
    """Write a sidecar JSON file in the prune_vocab_v2 schema. Returns path str."""
    if missing_keys:
        data = {"schema_version": 2}  # missing orig_to_new + new_to_orig
    else:
        if orig_to_new is None:
            # default: identity-like with a few overrides for the unit-test
            # mapping (orig 100 → new 0, 200 → 1, …, 999 → new 9 fallback).
            orig_to_new = [-1] * 1000
            for ch_idx, orig in enumerate([100, 200, 300, 400, 500, 600, 700, 800]):
                orig_to_new[orig] = ch_idx  # a..h → new 0..7
            # 999 (other-char bucket) → drop (fallback)
        if new_to_orig is None:
            new_to_orig = [100, 200, 300, 400, 500, 600, 700, 800]
        data = {
            "schema_version": 2,
            "target_vocab": len(new_to_orig),
            "orig_vocab": len(orig_to_new),
            "orig_to_new": orig_to_new,
            "new_to_orig": new_to_orig,
        }
    p = tmp_path / "orig_to_new_token_ids.json"
    p.write_text(json.dumps(data))
    return str(p)


def _install_fake_tokenizer(monkeypatch, serve, fake_tok=None):
    """Replace the lazy HF tokenizer loader with a fake."""
    fake = fake_tok or _FakeTokenizer()

    def _fake_get(state):
        state["tokenizer"] = fake
        state["tokenizer_tried"] = True
        # mirror the EOS-fallback refinement step in production
        orig_to_new = state["orig_to_new"]
        eos = getattr(fake, "eos_token_id", None)
        if eos is not None and 0 <= eos < len(orig_to_new) and orig_to_new[eos] >= 0:
            state["fallback_new_id"] = orig_to_new[eos]
        return fake

    monkeypatch.setattr(serve, "_get_remap_tokenizer", _fake_get)
    return fake


def test_route_request_no_remap_when_env_unset(monkeypatch):
    """Baseline: when VLLM_VOCAB_REMAP_SIDECAR is unset, route_request is
    unchanged — chat goes to /v1/chat/completions with messages, completion
    goes to /v1/completions with prompt. No prompt_token_ids, no logprobs."""
    monkeypatch.delenv("VLLM_VOCAB_REMAP_SIDECAR", raising=False)
    serve = load_serve_module()
    path, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        "/invocations",
    )
    assert path == "/v1/chat/completions"
    assert "messages" in payload
    assert "prompt_token_ids" not in payload
    assert "logprobs" not in payload
    assert "return_tokens_as_token_ids" not in payload


def test_route_request_with_remap_collapses_chat_to_completions(monkeypatch, tmp_path):
    """Remap path collapses chat → /v1/completions with prompt_token_ids."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    path, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "ab"}], "max_tokens": 32},
        "/invocations",
    )
    assert path == "/v1/completions"
    assert "prompt_token_ids" in payload
    assert "messages" not in payload
    # vLLM 0.19 OpenAI completions schema requires `prompt` field presence
    # (as a string) even when prompt_token_ids is the actual input — server
    # validates field presence before dispatching on which one to use.
    # Empty-string convention when prompt_token_ids carries the real input.
    assert payload.get("prompt") == ""


def test_route_request_with_remap_remaps_input_ids(monkeypatch, tmp_path):
    """Remap path applies orig_to_new. Fake tokenizer emits 100, 200 for 'ab',
    and the sidecar maps orig 100→new 0, 200→new 1, etc. — payload must carry
    new-space IDs [0, 1]."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    _, payload = serve.route_request(
        {"prompt": "ab", "max_tokens": 4}, "/v1/completions",
    )
    assert payload["prompt_token_ids"] == [0, 1]


def test_route_request_with_remap_includes_logprobs_and_return_tokens_as_token_ids(
        monkeypatch, tmp_path):
    """Silent-failure fix: vLLM only populates choices[].logprobs.token_ids when
    `logprobs>=1` AND `return_tokens_as_token_ids=true` are in the request.
    Without these, the response remap silently falls back to vLLM's wrong-
    tokenizer text. These flags MUST be in every remapped payload."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    _, payload = serve.route_request(
        {"prompt": "ab", "max_tokens": 4}, "/v1/completions",
    )
    assert payload.get("logprobs") == 1
    assert payload.get("return_tokens_as_token_ids") is True


def test_route_request_with_remap_preserves_default_stops(monkeypatch, tmp_path):
    """Caller stops MUST be merged with serve.py's DEFAULT_STOP_STRINGS (the
    "Wait, let me re" / "Wait, actually" self-correction phrases) even on the
    vocab-remap path, matching the non-remap chat path."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    monkeypatch.delenv("VLLM_DISABLE_THINK_STOPS", raising=False)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    _, payload = serve.route_request(
        {"prompt": "ab", "stop": ["</answer>"], "max_tokens": 4},
        "/v1/completions",
    )
    stops = payload["stop"]
    assert "</answer>" in stops, "caller stop missing"
    for default in serve.DEFAULT_STOP_STRINGS:
        assert default in stops, f"default stop {default!r} missing"


def test_route_request_with_remap_preserves_repetition_penalty(
        monkeypatch, tmp_path):
    """repetition_penalty MUST be present in the remapped payload, matching the
    non-remap path."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    monkeypatch.setenv("VLLM_REPETITION_PENALTY", "1.10")
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    _, payload = serve.route_request(
        {"prompt": "ab", "max_tokens": 4}, "/v1/completions",
    )
    assert payload.get("repetition_penalty") == pytest.approx(1.10)


def test_response_remap_inverts_token_ids(monkeypatch, tmp_path):
    """Response remap pulls choices[].logprobs.token_ids (new-space), maps via
    new_to_orig, and decodes locally. The decoded text must use orig-space
    token strings (the fake tokenizer encodes orig IDs as 'oNNN|oMMM|…')."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    # vLLM response with new-space token IDs in logprobs.token_ids:
    raw_resp = json.dumps({
        "object": "text_completion",
        "choices": [{
            "index": 0,
            "text": "wrong-tokenizer text",  # vLLM's wrong decode
            "logprobs": {"token_ids": [0, 1, 2]},  # new IDs → orig 100, 200, 300
            "finish_reason": "stop",
        }],
    }).encode()
    state = serve._get_vocab_remap_state()
    assert state is not None
    out = serve._remap_response_bytes(raw_resp, state)
    parsed = json.loads(out)
    text = parsed["choices"][0]["text"]
    # Should contain the fake-decoder's orig-id stringification, NOT the raw
    # "wrong-tokenizer text" that vLLM produced.
    assert text == "o100|o200|o300"


def test_response_remap_handles_dropped_token_fallback(monkeypatch, tmp_path):
    """orig_to_new[100] = -1 (dropped) → tokenize maps to fallback (EOS in new
    space). The fake tokenizer's EOS is orig id 7, but we set orig_to_new[7]=2,
    so dropped-token fallback should be new id 2 after _install_fake_tokenizer."""
    # Custom sidecar: drop orig 100 (set to -1); map orig 7 (eos) → new 2;
    # 200 → 1 as before so 'ab' still has one mapped + one dropped.
    o2n = [-1] * 1000
    o2n[7] = 2   # EOS-in-orig → new 2 (this becomes fallback after tokenizer load)
    o2n[200] = 1
    n2o = [-1, 200, 7]  # new 0 unused, new 1 → 200, new 2 → 7
    sidecar = _write_sidecar(tmp_path, orig_to_new=o2n, new_to_orig=n2o)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    # 'ab' → orig [100, 200]. 100 is dropped → fallback new 2; 200 → 1.
    _, payload = serve.route_request(
        {"prompt": "ab", "max_tokens": 4}, "/v1/completions",
    )
    assert payload["prompt_token_ids"] == [2, 1]


def test_remap_disabled_when_sidecar_missing(monkeypatch, tmp_path):
    """Env points at a nonexistent file → warn + passthrough (don't crash). The
    state must be None so route_request takes the standard chat-completions path."""
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", str(tmp_path / "does_not_exist.json"))
    serve = load_serve_module()
    state = serve._get_vocab_remap_state()
    assert state is None
    # Routing should fall back to the standard chat path.
    path, payload = serve.route_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        "/invocations",
    )
    assert path == "/v1/chat/completions"
    assert "prompt_token_ids" not in payload


def test_remap_disabled_when_sidecar_malformed(monkeypatch, tmp_path):
    """Sidecar JSON exists but missing required keys → warn + passthrough.
    Must NOT raise at module-init time."""
    sidecar = _write_sidecar(tmp_path, missing_keys=True)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    state = serve._get_vocab_remap_state()
    assert state is None
    # Module is still usable; standard routing must still work.
    path, _ = serve.route_request(
        {"prompt": "x", "max_tokens": 1}, "/v1/completions",
    )
    assert path == "/v1/completions"


def test_response_remap_handles_tokens_string_form(monkeypatch, tmp_path):
    """Some vLLM versions expose `logprobs.tokens` as a list of strings of the
    form 'token_id:NNN' instead of `logprobs.token_ids`. The extractor must
    parse both shapes — the audit fix requires testing both."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    raw_resp = json.dumps({
        "object": "text_completion",
        "choices": [{
            "index": 0,
            "text": "garbled",
            "logprobs": {"tokens": ["token_id:0", "token_id:1"]},  # new 0, 1
        }],
    }).encode()
    state = serve._get_vocab_remap_state()
    out = serve._remap_response_bytes(raw_resp, state)
    parsed = json.loads(out)
    assert parsed["choices"][0]["text"] == "o100|o200"


def test_response_remap_rewrites_chat_message_content(monkeypatch, tmp_path):
    """For chat-shaped vLLM responses (choices[].message.content), the remap
    must overwrite message.content too — not just choices[].text."""
    sidecar = _write_sidecar(tmp_path)
    monkeypatch.setenv("VLLM_VOCAB_REMAP_SIDECAR", sidecar)
    serve = load_serve_module()
    _install_fake_tokenizer(monkeypatch, serve)
    raw_resp = json.dumps({
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "garbled"},
            "logprobs": {"token_ids": [0, 1]},
        }],
    }).encode()
    state = serve._get_vocab_remap_state()
    out = serve._remap_response_bytes(raw_resp, state)
    parsed = json.loads(out)
    assert parsed["choices"][0]["message"]["content"] == "o100|o200"
