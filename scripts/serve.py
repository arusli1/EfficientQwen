#!/usr/bin/env python3
"""EfficientQwen serving entrypoint (Phase 1+).

Replaces the base image's serve_default.py. Same external contract — same
/ping, /invocations smart router, /v1/completions, /v1/chat/completions —
but launches vLLM with our tuning flags driven by VLLM_* environment vars
(see experiments/cyankiwi/config.env for the canonical list).

Architecture:
    eval host (port 8080)
       │
       └─ this script (HTTP router)
            └─ vLLM OpenAI server on 127.0.0.1:8081

For Phase 0 (exact 2.10x reference reproduction), use the base image's
ENTRYPOINT directly — don't build with this serve.py.
"""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ─── Constants ────────────────────────────────────────────────────────────────
# VLLM_INTERNAL_PORT exists for tests; production should leave default (8081)
INTERNAL_VLLM_PORT = int(os.environ.get("VLLM_INTERNAL_PORT", "8081"))
MODEL_DIR = os.environ.get("VLLM_MODEL", "/opt/ml/model")
EXTERNAL_PORT = int(os.environ.get("VLLM_PORT", "8080"))
SERVED_MODEL_NAME = "default"
# Default 590s leaves 10s headroom under the 600s submission /ping deadline.
# Local iteration with cold caches (first KV-fp8 / FlashInfer JIT compile)
# routinely exceeds this — overridable via env so cloud builds keep the tight
# default but local experiments can use a longer wait.
VLLM_HEALTH_TIMEOUT_S = int(os.environ.get("VLLM_HEALTH_TIMEOUT_S", "590"))

_vllm_ready = False
_vllm_proc: subprocess.Popen | None = None


# ─── Vocab-prune remap state (optional)
# Activated when VLLM_VOCAB_REMAP_SIDECAR points at a readable JSON file with
# schema produced by scripts/prune_vocab_v2.py (build_sidecar):
#     {orig_to_new: list[int], new_to_orig: list[int],
#      target_vocab, orig_vocab, schema_version}
# When active:
#   - inbound chat/completion requests are tokenized locally with the original
#     full-vocab HF tokenizer (loaded lazily), each ID remapped via orig_to_new
#     (dropped IDs fall back to EOS-in-new-space or new id 0), and sent to vLLM
#     via /v1/completions as prompt_token_ids — bypassing vLLM's tokenizer.
#   - outbound responses' per-choice token_ids (from logprobs.token_ids or
#     logprobs.tokens with "token_id:NNN") are remapped via new_to_orig and
#     decoded locally with the original tokenizer to overwrite choices[].text /
#     choices[].message.content.
#   - the request payload includes "logprobs": 1 and
#     "return_tokens_as_token_ids": true so vLLM populates per-choice token IDs
#     (without these flags the remap silently uses vLLM's wrong-tokenizer text).
# When VLLM_VOCAB_REMAP_SIDECAR is unset/empty: behaviour is unchanged.
_vocab_remap_state: dict | None = None  # populated by _get_vocab_remap_state()
_vocab_remap_init_done = False  # ensure load attempt happens at most once


# ─── Env-driven vLLM flag assembly ────────────────────────────────────────────
def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, default)
    return v if v not in (None, "") else None


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes")


def build_vllm_cmd() -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(INTERNAL_VLLM_PORT),
        "--gpu-memory-utilization", _env("VLLM_GPU_MEMORY_UTILIZATION", "0.92"),
        "--max-model-len", _env("VLLM_MAX_MODEL_LEN", "8448"),
        "--max-num-seqs", _env("VLLM_MAX_NUM_SEQS", "1"),
        # 2048 is vLLM's recommended chunked-prefill chunk size for ITL.
        # An 8K prompt becomes 4 x 2048 chunks interleaved with decode.
        "--max-num-batched-tokens", _env("VLLM_MAX_NUM_BATCHED_TOKENS", "2048"),
        "--block-size", _env("VLLM_BLOCK_SIZE", "16"),
        "--dtype", _env("VLLM_DTYPE", "float16"),
    ]

    # Quantization: omit the flag when unset to let vLLM auto-detect from
    # config.json. cyankiwi/Qwen3.5-4B-AWQ-4bit is misleadingly named — its
    # quant_method is "compressed-tensors" (llmcompressor), NOT awq. vLLM's
    # CompressedTensorsWNA16 backend internally dispatches to MarlinLinearKernel,
    # giving the same speed as awq_marlin would. Forcing --quantization awq_marlin
    # on compressed-tensors weights crashes the load. Only pass the flag if the
    # caller knows the weight format demands an override.
    if v := _env("VLLM_QUANTIZATION"):
        cmd += ["--quantization", v]

    if _env_bool("VLLM_LANGUAGE_MODEL_ONLY"):
        cmd.append("--language-model-only")
    if _env_bool("VLLM_ENABLE_CHUNKED_PREFILL"):
        cmd.append("--enable-chunked-prefill")
    if _env_bool("VLLM_ENABLE_PREFIX_CACHING"):
        cmd.append("--enable-prefix-caching")
    if _env_bool("VLLM_ENFORCE_EAGER"):
        cmd.append("--enforce-eager")
    if _env_bool("VLLM_TRUST_REMOTE_CODE"):
        cmd.append("--trust-remote-code")

    if v := _env("VLLM_KV_CACHE_DTYPE"):
        cmd += ["--kv-cache-dtype", v]
    if v := _env("VLLM_SPECULATIVE_CONFIG"):
        cmd += ["--speculative-config", v]
    if v := _env("VLLM_CUDAGRAPH_CAPTURE_SIZES"):
        # vLLM 0.19 actual flag: --cudagraph-capture-sizes (one word "cudagraph";
        # NOT --cuda-graph-sizes — that flag does not exist). Argparse signature is
        # `nargs="+"`, so each size must be a separate argv entry — split the
        # CSV value here. Passing "3,6,9,12,24" as one arg errors with
        # "invalid int value: '3,6,9,12,24'".
        cmd += ["--cudagraph-capture-sizes", *[s.strip() for s in v.split(",") if s.strip()]]
    if v := _env("VLLM_NUM_GPU_BLOCKS_OVERRIDE"):
        # Skips ~10-30s of memory profiling at startup once we know the value
        cmd += ["--num-gpu-blocks-override", v]
    if v := _env("VLLM_COMPILATION_CONFIG"):
        cmd += ["--compilation-config", v]
    # vLLM 0.19.0 new knobs (NEW_IDEAS §A.2-A.3, §B.1):
    if v := _env("VLLM_PERFORMANCE_MODE"):
        # {"balanced","interactivity","throughput"}; default balanced.
        # "interactivity" is designed for low-latency single-stream at small
        # batch sizes — matches our baseline workload.
        cmd += ["--performance-mode", v]
    if v := _env("VLLM_OPTIMIZATION_LEVEL"):
        # {"O0","O1","O2","O3"}; default O2. O3 enables aggressive fusion
        # passes (norm_quant, act_quant, attn_quant, etc.).
        cmd += ["--optimization-level", v]
    if v := _env("VLLM_MAMBA_CACHE_MODE"):
        # {"all","align","none"}; vLLM default "none". For cyankiwi hybrid
        # mamba2+attention, switching to "none" may reduce per-page overhead
        # at single-stream. Discovered via CRITIQUE 2026-05-22 22:15 audit.
        cmd += ["--mamba-cache-mode", v]
    if v := _env("VLLM_KERNEL_CONFIG"):
        # JSON dict: see vllm/config/kernel.py KernelConfig. Notable knobs:
        # enable_flashinfer_autotune (bool), moe_backend (irrelevant — dense).
        # Example: VLLM_KERNEL_CONFIG='{"enable_flashinfer_autotune":true}'
        cmd += ["--kernel-config", v]
    if v := _env("VLLM_ATTENTION_CONFIG"):
        # JSON dict: see vllm/config/attention.py AttentionConfig. Notable:
        # backend ({"FLASH_ATTN","TREE_ATTN","FLASHINFER",...}),
        # use_prefill_decode_attention (bool — separate prefill/decode kernels),
        # flash_attn_max_num_splits_for_cuda_graph (int, default 32).
        # Example: VLLM_ATTENTION_CONFIG='{"backend":"TREE_ATTN"}'
        cmd += ["--attention-config", v]

    # VLLM_CHAT_TEMPLATE env override lets a variant point at a different
    # chat_template.jinja without needing a sibling weights/ directory full of
    # symlinks (e.g., for the shorter-CoT chat template variant — the chat
    # template lives at experiments/<variant>/chat_template.jinja while the
    # variant still loads cyankiwi's weights).
    chat_template_override = _env("VLLM_CHAT_TEMPLATE")
    if chat_template_override and os.path.isfile(chat_template_override):
        cmd += ["--chat-template", chat_template_override]
        _check_chat_template_hash(chat_template_override)
    else:
        chat_template = os.path.join(MODEL_DIR, "chat_template.jinja")
        if os.path.isfile(chat_template):
            cmd += ["--chat-template", chat_template]
            _check_chat_template_hash(chat_template)

    return cmd


def _check_chat_template_hash(template_path: str) -> None:
    """Warn if the chat template content differs from the last warm prewarm.

    A changed chat template invalidates the prefix cache (its prefix hash no
    longer matches the warm cache from a prior boot), which can add a large
    first-request latency. This check surfaces that mismatch at boot so an
    operator can rebake or accept the cold-prefill cost on first request.

    Stores the hash at /opt/ml/vllm_cache/chat_template_hash.txt. The
    file is updated after first successful prewarm() returns True (see
    main() below). Compares against current template content on each boot.
    """
    import hashlib
    try:
        content = open(template_path, "rb").read()
        h = hashlib.sha256(content).hexdigest()[:16]
        hash_file = os.environ.get(
            "VLLM_CHAT_TEMPLATE_HASH_FILE",
            "/opt/ml/vllm_cache/chat_template_hash.txt",
        )
        if os.path.isfile(hash_file):
            prior = open(hash_file).read().strip()[:16]
            if prior and prior != h:
                print(
                    f"[serve] WARN chat-template hash mismatch: "
                    f"current={h} vs cached={prior}. First request may "
                    f"pay re-prefill cost (template changed since last "
                    f"warm boot).",
                    flush=True,
                )
            else:
                print(f"[serve] chat-template hash OK: {h}", flush=True)
        else:
            print(
                f"[serve] chat-template hash: {h} (no prior cache hash on "
                f"disk; will be written after first prewarm completes)",
                flush=True,
            )
    except Exception as e:
        print(f"[serve] chat-template hash check failed (non-fatal): {e}",
              flush=True)


def _record_chat_template_hash() -> None:
    """Persist current chat template hash after successful prewarm.
    Called from main() in the ready-thread once prewarm() returns True."""
    import hashlib
    try:
        chat_template_override = _env("VLLM_CHAT_TEMPLATE")
        if chat_template_override and os.path.isfile(chat_template_override):
            template_path = chat_template_override
        else:
            template_path = os.path.join(MODEL_DIR, "chat_template.jinja")
        if not os.path.isfile(template_path):
            return
        content = open(template_path, "rb").read()
        h = hashlib.sha256(content).hexdigest()[:16]
        hash_file = os.environ.get(
            "VLLM_CHAT_TEMPLATE_HASH_FILE",
            "/opt/ml/vllm_cache/chat_template_hash.txt",
        )
        os.makedirs(os.path.dirname(hash_file), exist_ok=True)
        with open(hash_file, "w") as f:
            f.write(h + "\n")
        print(f"[serve] chat-template hash recorded: {h} → {hash_file}",
              flush=True)
    except Exception as e:
        print(f"[serve] chat-template hash record failed (non-fatal): {e}",
              flush=True)


# ─── vLLM lifecycle ───────────────────────────────────────────────────────────
# PARO quantization bootstrap. When VLLM_USE_PAROQUANT=1, swap the vllm
# launch from `python -m vllm.entrypoints.openai.api_server …` to a small
# `python -c "…bootstrap…"` that imports paroquant.inference.backends.vllm,
# calls register() so the "paroquant" QuantizationConfig is in the registry
# BEFORE vllm parses config.json, then runs the openai api_server module via
# runpy with alter_sys=True so argparse picks up the same CLI args.
_PAROQUANT_BOOTSTRAP = (
    "import paroquant.inference.backends.vllm as _pq; _pq.register(); "
    "import runpy; "
    "runpy.run_module('vllm.entrypoints.openai.api_server', "
    "run_name='__main__', alter_sys=True)"
)


def start_vllm() -> subprocess.Popen:
    cmd = build_vllm_cmd()
    if _env_bool("VLLM_USE_PAROQUANT"):
        # Replace ['python','-m','vllm.entrypoints.openai.api_server',…args]
        # with ['python','-c',bootstrap,…args].
        cmd = [sys.executable, "-c", _PAROQUANT_BOOTSTRAP, *cmd[3:]]
    print(f"[serve] launch: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def wait_for_vllm_health(timeout: int = VLLM_HEALTH_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{INTERNAL_VLLM_PORT}/health", timeout=2,
            )
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def prewarm() -> bool:
    """Trigger Triton/CUDA-graph + chat-template prefix-cache warmup.

    Six requests hit the actual eval shapes BEFORE /ping flips to 200, so
    the first real eval request lands on a warm path. Covers:

    1-2. /v1/completions at 64 + 8192 prompt tokens — populates the bench-
         latency cudagraph shapes (short + long).
    3-4. /v1/chat/completions with enable_thinking=False (MMLU-Pro / IFEval
         shape) — populates chat-template prefix cache for non-thinking
         requests; this prefix is HOT for every quality eval request.
    5-6. /v1/chat/completions with enable_thinking=True (GPQA-D shape) —
         populates the thinking-mode prefix prelude.

    Why this matters: without warming the chat path, every first request
    pays a cold-prefill cost (~30-150ms on chat-completions) because the
    chat-template prefix is not yet cached. Disabled by
    VLLM_DISABLE_EXTENDED_PREWARM=1.
    """
    base_url = f"http://127.0.0.1:{INTERNAL_VLLM_PORT}"

    def _post(path: str, body: dict, label: str) -> bool:
        req = urllib.request.Request(
            f"{base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=180).read()
            return True
        except Exception as e:
            print(f"[serve] prewarm {label} failed: {e}", flush=True)
            return False

    # 1-2: /v1/completions at short + long shapes (populates bench cudagraphs)
    for length_hint in (1, 8):
        ok = _post("/v1/completions", {
            "model": SERVED_MODEL_NAME,
            "prompt": "Hi " * length_hint,
            "max_tokens": 4,
            "temperature": 0.0,
        }, f"completions len={length_hint}")
        if not ok:
            return False

    # Extended chat-template prewarm (default ON; toggle off for legacy probes)
    if os.environ.get("VLLM_DISABLE_EXTENDED_PREWARM", "0") in ("1", "true", "yes"):
        return True

    # 3-4: /v1/chat/completions thinking-off (MMLU/IFEval shape)
    for shape_msg in ("Pick one: A) yes B) no", "Write a short greeting."):
        ok = _post("/v1/chat/completions", {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": shape_msg}],
            "max_tokens": 4,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }, f"chat think=off len={len(shape_msg)}")
        if not ok:
            return False

    # 5-6: /v1/chat/completions thinking-on (GPQA-D shape) — short max_tokens
    # to avoid blowing prewarm wall budget on a thinking-mode generation
    for shape_msg in ("What is 2+2?", "Is the sky blue? Answer yes or no."):
        ok = _post("/v1/chat/completions", {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": shape_msg}],
            "max_tokens": 8,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": True},
        }, f"chat think=on len={len(shape_msg)}")
        if not ok:
            return False

    return True


def _proxy(path: str, payload: bytes) -> bytes:
    req = urllib.request.Request(
        f"http://127.0.0.1:{INTERNAL_VLLM_PORT}{path}",
        data=payload, headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=600).read()


def _proxy_get(path: str) -> bytes:
    return urllib.request.urlopen(
        f"http://127.0.0.1:{INTERNAL_VLLM_PORT}{path}", timeout=10,
    ).read()


# ─── Vocab-prune sidecar load + tokenizer + remap helpers ────────────────────
def _get_vocab_remap_state() -> dict | None:
    """Return remap state dict {orig_to_new, new_to_orig, fallback_new_id,
    sidecar_path} or None if the env switch is unset / sidecar is missing or
    malformed.

    The state is computed at most once per process (sidecar contents are static)
    but is re-evaluated on every call from `_get_vocab_remap_state` test paths
    that monkeypatch the env — so tests can flip the switch by clearing the
    cache via `_reset_vocab_remap_cache()`.
    """
    global _vocab_remap_state, _vocab_remap_init_done
    if _vocab_remap_init_done:
        return _vocab_remap_state
    _vocab_remap_init_done = True

    path = os.environ.get("VLLM_VOCAB_REMAP_SIDECAR", "").strip()
    if not path:
        _vocab_remap_state = None
        return None
    if not os.path.isfile(path):
        print(
            f"[serve] WARN VLLM_VOCAB_REMAP_SIDECAR={path!r} not found; "
            "passthrough (no remap)", flush=True,
        )
        _vocab_remap_state = None
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        orig_to_new = list(data["orig_to_new"])
        new_to_orig = list(data["new_to_orig"])
    except Exception as e:
        print(
            f"[serve] WARN failed to load VLLM_VOCAB_REMAP_SIDECAR={path!r} "
            f"({e}); passthrough (no remap)", flush=True,
        )
        _vocab_remap_state = None
        return None

    _vocab_remap_state = {
        "orig_to_new": orig_to_new,
        "new_to_orig": new_to_orig,
        "fallback_new_id": 0,  # refined to EOS-in-new-space after tokenizer load
        "sidecar_path": path,
        "tokenizer": None,  # lazy
        "tokenizer_tried": False,
    }
    print(
        f"[serve] vocab-prune remap ACTIVE: sidecar={path} "
        f"|orig_to_new|={len(orig_to_new)} |new_to_orig|={len(new_to_orig)}",
        flush=True,
    )
    return _vocab_remap_state


def _reset_vocab_remap_cache() -> None:
    """Test hook: force re-evaluation of the env switch on next access."""
    global _vocab_remap_state, _vocab_remap_init_done
    _vocab_remap_state = None
    _vocab_remap_init_done = False


def _get_remap_tokenizer(state: dict):
    """Load the HF tokenizer lazily from VLLM_MODEL. Returns the tokenizer or
    None if load failed (e.g., transformers unavailable in tests).

    Tokenizer load is deferred until the first request that needs it, so module
    import does not require transformers when the env var is unset.
    """
    if state["tokenizer_tried"]:
        return state["tokenizer"]
    state["tokenizer_tried"] = True
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    except Exception as e:
        print(f"[serve] WARN tokenizer load for remap failed: {e}", flush=True)
        state["tokenizer"] = None
        return None
    state["tokenizer"] = tok
    # Refine the dropped-token fallback to EOS-in-new-space (safer than id 0).
    orig_to_new = state["orig_to_new"]
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None and 0 <= eos < len(orig_to_new) and orig_to_new[eos] >= 0:
        state["fallback_new_id"] = orig_to_new[eos]
        print(
            f"[serve] remap dropped-token fallback new_id={state['fallback_new_id']} "
            f"(orig EOS={eos})", flush=True,
        )
    return tok


def _o2n(state: dict, orig_id: int) -> int:
    """Map original token ID to pruned-vocab ID. Out-of-range or dropped → fallback."""
    orig_to_new = state["orig_to_new"]
    if not (0 <= orig_id < len(orig_to_new)):
        return state["fallback_new_id"]
    new = orig_to_new[orig_id]
    return new if new >= 0 else state["fallback_new_id"]


def _n2o(state: dict, new_id: int) -> int:
    """Map pruned-vocab ID back to original. Out-of-range → passthrough."""
    new_to_orig = state["new_to_orig"]
    if not (0 <= new_id < len(new_to_orig)):
        return new_id
    o = new_to_orig[new_id]
    return o if o >= 0 else new_id


def _render_remap_prompt(state: dict, data: dict, path: str) -> tuple[list[int], bool]:
    """Render the request's prompt into the pruned token-ID space.

    Returns (new_ids, was_chat). For chat: applies the model's chat template
    locally to get the rendered text. For completion: uses data['prompt']
    (unwrapping list form).
    """
    use_chat = "messages" in data and path != "/v1/completions"
    tok = _get_remap_tokenizer(state)
    if tok is None:
        raise RuntimeError("vocab-prune remap requires HF tokenizer; load failed")
    if use_chat:
        thinking = bool(data.get("thinking", False)) or bool(
            data.get("chat_template_kwargs", {}).get("enable_thinking", False)
        )
        rendered = tok.apply_chat_template(
            data["messages"],
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": thinking},
        )
    else:
        rendered = data.get("prompt", "")
        if isinstance(rendered, list):
            rendered = rendered[0] if rendered else ""
    # add_special_tokens=False — chat template injected them; raw completions
    # mirror vLLM's default (Qwen doesn't add BOS).
    orig_ids = list(tok.encode(rendered, add_special_tokens=False))
    new_ids = [_o2n(state, i) for i in orig_ids]
    return new_ids, use_chat


def _extract_token_ids(choice: dict) -> list[int]:
    """Pull per-token IDs from a vLLM choice. vLLM 0.19 with
    `return_tokens_as_token_ids=true` exposes them in either
    `logprobs.token_ids` (list[int]) or as string-encoded
    `logprobs.tokens` entries of the form "token_id:NNN" — handle both.
    Returns empty list if neither shape is present.
    """
    lp = choice.get("logprobs") or {}
    ids = lp.get("token_ids")
    if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
        return list(ids)
    tokens = lp.get("tokens")
    if isinstance(tokens, list):
        out: list[int] = []
        for t in tokens:
            if isinstance(t, str) and t.startswith("token_id:"):
                try:
                    out.append(int(t.split(":", 1)[1]))
                except ValueError:
                    pass
        if out:
            return out
    return []


def _remap_response_bytes(raw: bytes, state: dict) -> bytes:
    """Decode vLLM's pruned-space token IDs back to original-space text.

    Rewrites choices[].text (completion shape) and
    choices[].message.content (chat shape) using the original HF tokenizer.
    Returns the original bytes unchanged on parse error.
    """
    try:
        resp = json.loads(raw)
    except Exception:
        return raw
    tok = _get_remap_tokenizer(state)
    if tok is None:
        return raw
    changed = False
    for choice in resp.get("choices", []) or []:
        ids = _extract_token_ids(choice)
        if not ids:
            continue
        orig_ids = [_n2o(state, i) for i in ids]
        text = tok.decode(orig_ids, skip_special_tokens=True)
        if "message" in choice and isinstance(choice["message"], dict):
            choice["message"]["content"] = text
        # Always overwrite text (some chat responses also expose it).
        choice["text"] = text
        changed = True
    if not changed:
        return raw
    return json.dumps(resp).encode()


# ─── Pure routing logic (extracted for testability) ───────────────────────────
# Reserved tokens at the END of max_model_len for the rendered prompt + chat
# template overhead. vLLM rejects requests where (prompt + max_tokens) exceeds
# max_model_len. We cap max_tokens to (max_model_len - prompt budget) so the
# caller-supplied max_tokens never causes a 400. 1024 is conservative for
# quality-eval prompts (typical: 200-700 tokens including system + few-shot).
_PROMPT_BUDGET_TOKENS = 1024


def _cap_max_tokens(requested: int, max_model_len: int) -> int:
    """Clamp max_tokens so the request never exceeds vLLM's max_model_len.

    Floor of 64 keeps the floor sane if max_model_len is tiny (test setups).
    """
    safe = max(64, max_model_len - _PROMPT_BUDGET_TOKENS)
    return min(int(requested), safe)


# B6: static stop sequences applied to ALL chat-completions requests.
# Targets explicit self-correction phrases that often precede truncated mid-
# reasoning answers in thinking-mode generation. Static + task-agnostic →
# NOT benchmark detection per docs/STATE.md:64. Disable via
# VLLM_DISABLE_THINK_STOPS=1 for A/B testing. Empty list when disabled.
DEFAULT_STOP_STRINGS = [
    # Empirically chosen from frequency analysis on I5 GPQA samples:
    # "wait, let me" appears in 13.1% of responses (usually self-correction:
    # "wait, let me reconsider/recheck/re-read"); "wait, actually" in 3.0%.
    # Earlier hand-picked phrases (e.g., "I made an error") had 0% hit rate
    # → useless. These match what the model actually says.
    "Wait, let me re",
    "Wait, actually",
]


def _default_stops() -> list[str]:
    return [] if _env_bool("VLLM_DISABLE_THINK_STOPS") else list(DEFAULT_STOP_STRINGS)


def route_request(data: dict, path: str) -> tuple[str, dict]:
    """Translate an incoming /invocations request into a vLLM call.

    Returns (vllm_path, vllm_payload_dict). Pure function (apart from the
    lazy-loaded tokenizer side-effect on the vocab-remap path); no network I/O.
    Reads VLLM_MAX_MODEL_LEN from env at call time (so test overrides work).
    """
    use_chat = "messages" in data and path != "/v1/completions"
    thinking = bool(data.get("thinking", False)) or bool(
        data.get("chat_template_kwargs", {}).get("enable_thinking", False)
    )
    max_model_len = int(_env("VLLM_MAX_MODEL_LEN", "8448"))

    # Vocab-prune remap path: tokenize locally and collapse both chat and
    # completion requests to /v1/completions with prompt_token_ids in the
    # pruned vocab space. Keeps vLLM out of the tokenizer entirely.
    remap_state = _get_vocab_remap_state()
    if remap_state is not None:
        new_ids, _was_chat = _render_remap_prompt(remap_state, data, path)
        if use_chat:
            default_max = 12288 if thinking else 128
        else:
            default_max = 128
        requested_max = data.get("max_tokens", default_max)
        client_stops = list(data.get("stop") or [])
        payload = {
            "model": SERVED_MODEL_NAME,
            # vLLM 0.19 OpenAI completions schema requires `prompt` as a
            # string even when `prompt_token_ids` is the actual input —
            # the server validates field presence before checking which to
            # use. Bench requests failed without this; empty string is
            # treated as "no text prompt" and the token IDs path takes
            # over. (1-line unblock for v64k bench.)
            "prompt": "",
            "prompt_token_ids": new_ids,
            "max_tokens": _cap_max_tokens(requested_max, max_model_len),
            "temperature": data.get("temperature", 0.0),
            # Default stops merged with caller stops (same as the
            # non-remap chat path). Stops are strings — vLLM matches against
            # the decoded text, which it produces using its (wrong) tokenizer.
            # The local re-decode in _remap_response_bytes is what the client
            # sees; vLLM-side stop matching is best-effort under remap.
            "stop": client_stops + _default_stops(),
            # Repetition penalty from env (parity with the non-remap path).
            "repetition_penalty": _rep_penalty(),
            # Silent-failure fix: force vLLM to expose per-token IDs so the
            # response remap can decode locally instead of trusting vLLM's
            # wrong-tokenizer text. Without these, choices[].logprobs is empty
            # and the remap silently passes through garbled text.
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        }
        return "/v1/completions", payload

    if use_chat:
        default_max = 12288 if thinking else 128
        requested_max = data.get("max_tokens", default_max)
        # Merge client-supplied stops with our static default set.
        client_stops = list(data.get("stop") or [])
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": data["messages"],
            "max_tokens": _cap_max_tokens(requested_max, max_model_len),
            "temperature": data.get("temperature", 0.0),
            "chat_template_kwargs": {"enable_thinking": thinking},
            "stop": client_stops + _default_stops(),
            # B7: mild repetition penalty to suppress degenerate output loops
            # (subagent H found 14% of GPQA responses had 50-char chunks
            # repeated ≥4×, 12/14 of those wrong). Static + universal config →
            # inference tuning, not benchmark detection (legal per STATE.md).
            # 1.05 is conservative; raise to 1.10 for stronger suppression.
            "repetition_penalty": _rep_penalty(),
        }
        return "/v1/chat/completions", payload

    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    requested_max = data.get("max_tokens", 128)
    return "/v1/completions", {
        "model": SERVED_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": _cap_max_tokens(requested_max, max_model_len),
        "temperature": data.get("temperature", 0.0),
        "repetition_penalty": _rep_penalty(),
    }


def _rep_penalty() -> float:
    """Reads VLLM_REPETITION_PENALTY env (default 1.05). Robust to empty
    string (which _env returns as None, breaking float() in the old code)."""
    raw = os.environ.get("VLLM_REPETITION_PENALTY", "").strip()
    if not raw:
        return 1.05
    try:
        return float(raw)
    except ValueError:
        return 1.05


# ─── HTTP router ──────────────────────────────────────────────────────────────
class _ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _vllm_alive() -> bool:
    """True if /ping should return 200 — vllm flagged ready AND subprocess alive."""
    if not _vllm_ready:
        return False
    # If subprocess exited, vLLM crashed mid-flight → /ping should flip to 503
    return not (_vllm_proc is not None and _vllm_proc.poll() is not None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200 if _vllm_alive() else 503)
            self.end_headers()
            return
        if self.path == "/v1/models":
            # lm-eval-harness probes this on init. Proxy to vLLM, which serves
            # a real /v1/models. Fall back to a canned shape if proxying fails.
            try:
                result = _proxy_get("/v1/models")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(result)
                return
            except Exception:
                canned = json.dumps({
                    "object": "list",
                    "data": [{"id": SERVED_MODEL_NAME, "object": "model",
                              "owned_by": "local"}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(canned)
                return
        if self.path == "/metrics":
            # vLLM 0.19.0 exposes Prometheus-style /metrics on its OpenAI api
            # server. scripts/profile_model.py + scripts/bench_latency.py both
            # query this for per-phase attribution + MTP-acceptance counters
            # (vllm:e2e_request_latency_seconds_sum, vllm:spec_decode_*).
            # Without this proxy, those scripts silently fall back to the
            # broken coarse model (token-count-ratio instead of time-ratio).
            try:
                result = _proxy_get("/metrics")
                self.send_response(200)
                # Prometheus exposition format is plain text, not JSON.
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(result)
                return
            except Exception:
                # If the internal vLLM /metrics isn't up yet (cold-start), 503.
                # Callers (profile_model.py) handle this gracefully via
                # try/except around the request.
                self.send_response(503)
                self.end_headers()
                return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/invocations", "/v1/completions", "/v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return

        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            data = json.loads(body)
            vllm_path, vllm_payload = route_request(data, self.path)
            result = _proxy(vllm_path, json.dumps(vllm_payload).encode())
            # When vocab-prune remap is active, rewrite per-choice text using
            # the original tokenizer so the client sees orig-space text.
            remap_state = _get_vocab_remap_state()
            if remap_state is not None:
                result = _remap_response_bytes(result, remap_state)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(result)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    global _vllm_ready, _vllm_proc

    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        print(f"[serve] FATAL: no config.json at {MODEL_DIR}", flush=True)
        return 1

    _vllm_proc = start_vllm()

    def _ready_thread():
        global _vllm_ready
        if not wait_for_vllm_health():
            print("[serve] vLLM health timeout; /ping stays 503", flush=True)
            return
        if not prewarm():
            print("[serve] prewarm failed; /ping stays 503", flush=True)
            return
        # Record chat-template hash so the next boot can detect changes
        _record_chat_template_hash()
        _vllm_ready = True
        print("[serve] /ping → 200", flush=True)

    threading.Thread(target=_ready_thread, daemon=True).start()

    print(f"[serve] listening on 0.0.0.0:{EXTERNAL_PORT}", flush=True)
    try:
        _ThreadingHTTPServer(("0.0.0.0", EXTERNAL_PORT), Handler).serve_forever()
    finally:
        # Make sure vLLM dies when we do
        if _vllm_proc is not None and _vllm_proc.poll() is None:
            _vllm_proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
