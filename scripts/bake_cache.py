#!/usr/bin/env python3
"""Pre-bake torch.compile cache at Docker build time (cold-start mitigation).

Launches vLLM with the same flags scripts/serve.py would use at runtime (driven
by VLLM_* env vars; cyankiwi config.env values are baked into the Dockerfile). Waits
for /health, runs one decode request to populate cudagraph capture for the
runtime batch shape (MTP=7 + 1 prompt token = 8), then cleanly terminates vLLM
and exits 0.

Goal: produce /opt/ml/vllm_cache/torch_compile_cache/<hash>/ inside the image
so target A10G cold-start skips the ~317s torch.compile work and hits the
warm-cache fast path (~156s total cold-start vs 482-697s without bake).

Cache portability requires the device-name spoof shim (scripts/_cache_patch.py)
to be active via PYTHONSTARTUP. The Dockerfile sets both ENV vars before this
RUN step.

Exit codes:
  0 — bake succeeded; cache populated
  1 — vLLM never reached /health within timeout
  2 — decode request failed (vLLM started but couldn't serve)
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VLLM_PORT = 8181  # internal port; doesn't conflict with the serve.py runtime 8081
HEALTH_TIMEOUT_S = 1200  # generous; this is build-time, not runtime
DECODE_TIMEOUT_S = 240


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else None


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes")


def _build_vllm_cmd() -> list[str]:
    """Mirror scripts/serve.py:build_vllm_cmd() so the bake matches runtime exactly.

    Keep in sync with scripts/serve.py — if either drifts, the bake's compile
    cache won't match runtime's cache key and we eat the cold-start anyway.
    """
    model = _env("VLLM_MODEL", "/opt/ml/model")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", "default",
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", _env("VLLM_GPU_MEMORY_UTILIZATION", "0.92"),
        "--max-model-len", _env("VLLM_MAX_MODEL_LEN", "13312"),
        "--max-num-seqs", _env("VLLM_MAX_NUM_SEQS", "1"),
        "--max-num-batched-tokens", _env("VLLM_MAX_NUM_BATCHED_TOKENS", "1024"),
        "--block-size", _env("VLLM_BLOCK_SIZE", "16"),
        "--dtype", _env("VLLM_DTYPE", "float16"),
    ]
    if _env_bool("VLLM_LANGUAGE_MODEL_ONLY"):
        cmd.append("--language-model-only")
    if _env_bool("VLLM_ENABLE_CHUNKED_PREFILL"):
        cmd.append("--enable-chunked-prefill")
    if _env_bool("VLLM_ENABLE_PREFIX_CACHING"):
        cmd.append("--enable-prefix-caching")
    if _env_bool("VLLM_ENFORCE_EAGER"):
        cmd.append("--enforce-eager")
    if v := _env("VLLM_QUANTIZATION"):
        cmd += ["--quantization", v]
    if v := _env("VLLM_SPECULATIVE_CONFIG"):
        cmd += ["--speculative-config", v]
    if v := _env("VLLM_CUDAGRAPH_CAPTURE_SIZES"):
        cmd += ["--cudagraph-capture-sizes",
                *[s.strip() for s in v.split(",") if s.strip()]]

    chat_template = Path(model) / "chat_template.jinja"
    if chat_template.is_file():
        cmd += ["--chat-template", str(chat_template)]
    return cmd


def _wait_for_health(deadline: float) -> bool:
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2,
            )
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _decode_once(prompt_repeat: int) -> bool:
    """Send one /v1/completions request to force cudagraph capture.

    prompt_repeat controls the prompt length (in 2-token chunks of "Hi ").
    """
    body = json.dumps({
        "model": "default",
        "prompt": "Hi " * prompt_repeat,
        "max_tokens": 4,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}/v1/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=DECODE_TIMEOUT_S).read()
        return True
    except Exception as e:
        print(f"[prebake] decode failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    if not Path(_env("VLLM_MODEL", "/opt/ml/model"), "config.json").is_file():
        print(f"[prebake] FATAL: no config.json at {_env('VLLM_MODEL', '/opt/ml/model')}",
              file=sys.stderr)
        return 1

    print(f"[prebake] launching vLLM on internal port {VLLM_PORT}", flush=True)
    cmd = _build_vllm_cmd()
    print(f"[prebake] cmd: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd)
    try:
        t0 = time.time()
        deadline = t0 + HEALTH_TIMEOUT_S
        if not _wait_for_health(deadline):
            print(f"[prebake] FAIL: vLLM not healthy within {HEALTH_TIMEOUT_S}s",
                  file=sys.stderr)
            return 1
        print(f"[prebake] vLLM healthy after {time.time() - t0:.1f}s", flush=True)

        # Two decodes to populate both common cudagraph shapes:
        #   prompt_repeat=1 → ~2 prompt tokens, batch shape 1 (prewarm path)
        #   prompt_repeat=4 → ~8 prompt tokens, batch shape 8 (MTP=7 + 1 = runtime hot path)
        print("[prebake] decode 1/2: short prompt (warms prewarm path)", flush=True)
        if not _decode_once(1):
            return 2
        print("[prebake] decode 2/2: 8-token prompt (warms MTP runtime path)", flush=True)
        if not _decode_once(4):
            return 2

        print("[prebake] cache populated; shutting down vLLM", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
