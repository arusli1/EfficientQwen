#!/usr/bin/env python3
"""Chat-completions latency profiler — catches what bench_latency.py misses.

`bench_latency.py` hits `/v1/completions`, which does NOT invoke the chat
template. The target host's medium-category latency benchmark hits
`/v1/chat/completions` with thinking-mode prompts, so a regression introduced
by the chat template (e.g. a system-prompt injection that breaks prefix
caching) is invisible to `bench_latency.py`.

This script is the missing pillar. Same API surface as `bench_latency.py`
(--runs, --warmup, --url, per-category JSON output) but POSTs to
`/v1/chat/completions` with the request shape the eval harness uses:
  messages=[{"role": "user", "content": <prompt>}]
  chat_template_kwargs={"enable_thinking": <per-cat default>}

Per-category thinking-mode defaults match the eval harness:
  short:  thinking=False (matches MMLU-Pro non-thinking shape)
  medium: thinking=False (matches IFEval non-thinking shape)
  long:   thinking=True  (matches GPQA-D thinking-on shape)

Override per category with --thinking-{short,medium,long} {on,off,auto}.

Hits the variant's external port (typically 8080 → routed through the
serve.py wrapper), so the same default stop strings, repetition penalty, and
chat_template_kwargs that serve.py injects are applied before the request
reaches vLLM. This matches the served path exactly.

Usage:
  python3 scripts/bench_chat_latency.py --runs 10 --warmup 3 \\
      --out experiments/<variant>/latency_chat_<date>.json

Output shape (schema-compatible with bench_latency.py outputs +
endpoint/thinking_mode discriminators):
  {
    "<cat>": {
      "median_ms": ..., "min_ms": ..., "max_ms": ..., "n": ...,
      "speedup_vs_baseline": ...,
      "mean_acceptance_length": ..., "n_drafts": ...,
      "endpoint": "chat",
      "thinking_mode": "on" | "off"
    },
    "overall": {"avg_median_ms": ..., "avg_speedup": ...,
                "endpoint": "chat"}
  }
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Same shapes as bench_latency.py for direct comparability
PROMPT_CONFIGS = {
    "short": {"num_tokens": 64, "max_new_tokens": 128, "thinking_default": False},
    "medium": {"num_tokens": 2048, "max_new_tokens": 256, "thinking_default": False},
    "long": {"num_tokens": 8192, "max_new_tokens": 256, "thinking_default": True},
}
BASELINE_MS = {"short": 2582, "medium": 5441, "long": 6576}


def _load_calib_prompts(path: Path) -> list[str]:
    """Return raw text prompts from calibration JSONL."""
    out: list[str] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("text") or obj.get("prompt") or obj.get("content") or ""
            if t and len(t.strip()) >= 20:
                out.append(t)
    return out


def _shape_prompt(text: str, target_tokens: int) -> str:
    """Adjust prompt to roughly target_tokens by repeating + truncating chars.

    Matches bench_latency.py's heuristic (4 chars ≈ 1 token for Qwen3.5 BPE).
    """
    target_chars = target_tokens * 4
    if len(text) < target_chars:
        sep = "\n\n---\n\n"
        rep = (text + sep) * (target_chars // (len(text) + len(sep)) + 1)
        return rep[:target_chars]
    return text[:target_chars]


def _invoke(url: str, prompt: str, max_tokens: int, thinking: bool,
            timeout: int = 600) -> float:
    """Send /v1/chat/completions, return wall ms.

    Caller-supplied chat_template_kwargs.enable_thinking is the gate for
    whether the chat template's thinking-mode branch fires. serve.py reads
    `data["chat_template_kwargs"]["enable_thinking"]` (see
    scripts/serve.py:route_request).
    """
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    urllib.request.urlopen(req, timeout=timeout).read()
    return (time.perf_counter() - t0) * 1000


def _fetch_spec_metrics(url: str) -> tuple[float, float]:
    """Pull (accepted_total, drafts_total) from /metrics."""
    try:
        text = urllib.request.urlopen(f"{url}/metrics", timeout=10).read().decode()
    except Exception:
        return (0.0, 0.0)
    pat = re.compile(r"^(vllm:spec_decode_\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE)
    m = {n: float(v) for n, v in pat.findall(text)}
    acc = m.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    drf = m.get("vllm:spec_decode_num_drafts_total", 0.0)
    return acc, drf


def _resolve_thinking(args, cat: str, cfg: dict) -> bool:
    """Resolve per-cat thinking flag: CLI override > per-cat default."""
    override = getattr(args, f"thinking_{cat}", "auto")
    if override == "on":
        return True
    if override == "off":
        return False
    return cfg["thinking_default"]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=os.environ.get("CONTAINER_URL", "http://localhost:8080"))
    p.add_argument("--runs", type=int, default=10,
                   help="measured runs per category (default 10)")
    p.add_argument("--warmup", type=int, default=3,
                   help="warmup runs per category (default 3)")
    _default_out = Path(os.environ["OUTPUT_PATH"]) if os.environ.get("OUTPUT_PATH") else None
    p.add_argument("--out", type=Path, default=_default_out)
    p.add_argument("--categories", default="short,medium,long")
    for cat in PROMPT_CONFIGS:
        p.add_argument(f"--thinking-{cat}", choices=["on", "off", "auto"], default="auto",
                       help=f"override thinking-mode for {cat} (default: auto = "
                            f"{PROMPT_CONFIGS[cat]['thinking_default']})")
    p.add_argument("--calib", type=Path,
                   default=Path("data/calibration_1024.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    prompts = _load_calib_prompts(args.calib)
    print(f"loaded {len(prompts)} calibration prompts", file=sys.stderr)
    assert len(prompts) >= 50, f"need >=50 prompts, got {len(prompts)}"
    random.shuffle(prompts)

    cats = [c.strip() for c in args.categories.split(",") if c.strip() in PROMPT_CONFIGS]
    results: dict = {}
    prompt_idx = 0

    for cat in cats:
        cfg = PROMPT_CONFIGS[cat]
        thinking = _resolve_thinking(args, cat, cfg)

        n_needed = args.warmup + args.runs
        shaped: list[str] = []
        for _ in range(n_needed):
            shaped.append(_shape_prompt(prompts[prompt_idx % len(prompts)], cfg["num_tokens"]))
            prompt_idx += 1

        # Warmup — populates chat-template prefix cache + cudagraphs, not
        # decode prefix (all prompts different).
        for w in range(args.warmup):
            with contextlib.suppress(Exception):
                _invoke(args.url, shaped[w], cfg["max_new_tokens"], thinking)

        acc_pre, drf_pre = _fetch_spec_metrics(args.url)

        latencies: list[float] = []
        for i in range(args.runs):
            try:
                ms = _invoke(args.url, shaped[args.warmup + i], cfg["max_new_tokens"],
                             thinking)
                latencies.append(ms)
                print(f"  [{cat} think={thinking}] run {i+1}/{args.runs}: {ms:.1f} ms",
                      file=sys.stderr)
            except Exception as e:
                print(f"  [{cat}] run {i+1} FAILED: {e}", file=sys.stderr)

        acc_post, drf_post = _fetch_spec_metrics(args.url)
        drafts_this = drf_post - drf_pre
        accepted_this = acc_post - acc_pre
        mean_acc = 1.0 + accepted_this / drafts_this if drafts_this > 0 else None

        if latencies:
            med = statistics.median(latencies)
            std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            results[cat] = {
                "median_ms": round(med, 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "std_ms": round(std, 2),
                "n": len(latencies),
                "speedup_vs_baseline": round(BASELINE_MS[cat] / med, 3),
                "mean_acceptance_length": round(mean_acc, 3) if mean_acc else None,
                "n_drafts": int(drafts_this),
                "endpoint": "chat",
                "thinking_mode": "on" if thinking else "off",
            }
            acc_str = f"{mean_acc:.2f}" if mean_acc else "n/a"
            print(f"  [{cat}] median {med:.1f}ms  speedup {BASELINE_MS[cat]/med:.3f}x  "
                  f"std {std:.1f}  accept_len {acc_str}  think={thinking}",
                  file=sys.stderr)

    if results:
        medians = [v["median_ms"] for v in results.values()]
        sp_avg = sum(v["speedup_vs_baseline"] for v in results.values()) / len(results)
        results["overall"] = {
            "avg_median_ms": round(statistics.mean(medians), 2),
            "avg_speedup": round(sp_avg, 3),
            "endpoint": "chat",
        }
        print(f"\noverall avg_speedup: {sp_avg:.3f}x  (chat-completions path)",
              file=sys.stderr)

    output = json.dumps(results, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
