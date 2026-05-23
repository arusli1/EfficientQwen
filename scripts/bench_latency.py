#!/usr/bin/env python3
"""Realistic latency profiler — uses diverse natural prompts, not filler.

The eval harness ranks latency on a private test set of natural prompts.
A filler-based profiler (e.g. "quick brown fox" repeated) is misleading because:

  1. It lets MTP achieve ~8/7 mean acceptance (trivially predictable)
  2. It hits ~70% prefix-cache hit rate by repeating the SAME prompt 5+ times
  3. It doesn't exercise the diverse prefill-token distributions that matter

This script instead samples DIVERSE prompts from the calibration corpus
(`data/calibration_1024.jsonl`) for each measured run. No prompt repeats
within a category. Token budgets match the eval host's spec:
  short:  64-tok prompt → 128 max_new_tokens
  medium: 2048-tok prompt → 256 max_new_tokens
  long:   8192-tok prompt → 256 max_new_tokens

Uses tokenized prompts padded/truncated to exact length for fair comparison
to the baseline.

Usage:
  python3 scripts/bench_latency.py --runs 5 --warmup 2 \\
      --out experiments/<variant>/latency_<date>.json
  python3 scripts/bench_latency.py --runs 5 --calib data/calibration_1024.jsonl

The Makefile invokes via `make eval-latency VARIANT=name`, which sets
OUTPUT_PATH env so the output lands in experiments/<variant>/ automatically.

Outputs:
- median_ms / min_ms / max_ms per category
- speedup_vs_baseline (using SAME 2582/5441/6576 baseline as competition spec)
- mean_acceptance_length captured from /metrics (delta across the run)
"""
from __future__ import annotations

import argparse
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

PROMPT_CONFIGS = {
    "short": {"num_tokens": 64, "max_new_tokens": 128},
    "medium": {"num_tokens": 2048, "max_new_tokens": 256},
    "long": {"num_tokens": 8192, "max_new_tokens": 256},
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

    We use a simple char-based heuristic (4 chars ≈ 1 token for Qwen3.5 BPE).
    Exact tokenization happens server-side; this is just to hit the right
    rough ballpark.
    """
    target_chars = target_tokens * 4
    if len(text) < target_chars:
        # Repeat with separator until we exceed target
        sep = "\n\n---\n\n"
        rep = (text + sep) * (target_chars // (len(text) + len(sep)) + 1)
        return rep[:target_chars]
    return text[:target_chars]


def _invoke(url: str, prompt: str, max_tokens: int, timeout: int = 600) -> float:
    """Send /v1/completions, return wall ms."""
    body = json.dumps({
        "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/completions",
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("CONTAINER_URL", "http://localhost:8080"))
    p.add_argument("--runs", type=int, default=5, help="measured runs per category")
    p.add_argument("--warmup", type=int, default=2)
    _default_out = Path(os.environ["OUTPUT_PATH"]) if os.environ.get("OUTPUT_PATH") else None
    p.add_argument("--out", type=Path, default=_default_out)
    p.add_argument("--categories", default="short,medium,long")
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
    results = {}
    prompt_idx = 0

    for cat in cats:
        cfg = PROMPT_CONFIGS[cat]

        # Pre-shape (warmup + measured) prompts for this category, ALL different
        n_needed = args.warmup + args.runs
        shaped: list[str] = []
        for _ in range(n_needed):
            shaped.append(_shape_prompt(prompts[prompt_idx % len(prompts)], cfg["num_tokens"]))
            prompt_idx += 1

        # Warmup (timed but discarded; populates compile + cudagraph but NOT
        # prefix cache because prompts are all different)
        import contextlib
        for w in range(args.warmup):
            with contextlib.suppress(Exception):
                _invoke(args.url, shaped[w], cfg["max_new_tokens"])

        # Spec-decode metrics: baseline BEFORE this category
        acc_pre, drf_pre = _fetch_spec_metrics(args.url)

        # Measured
        latencies: list[float] = []
        for i in range(args.runs):
            try:
                ms = _invoke(args.url, shaped[args.warmup + i], cfg["max_new_tokens"])
                latencies.append(ms)
                print(f"  [{cat}] run {i+1}/{args.runs}: {ms:.1f} ms", file=sys.stderr)
            except Exception as e:
                print(f"  [{cat}] run {i+1} FAILED: {e}", file=sys.stderr)

        acc_post, drf_post = _fetch_spec_metrics(args.url)
        drafts_this = drf_post - drf_pre
        accepted_this = acc_post - acc_pre
        mean_acc = 1.0 + accepted_this / drafts_this if drafts_this > 0 else None

        if latencies:
            med = statistics.median(latencies)
            results[cat] = {
                "median_ms": round(med, 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "n": len(latencies),
                "speedup_vs_baseline": round(BASELINE_MS[cat] / med, 3),
                "mean_acceptance_length": round(mean_acc, 3) if mean_acc else None,
                "n_drafts": int(drafts_this),
            }
            acc_str = f"{mean_acc:.2f}" if mean_acc else "n/a"
            print(f"  [{cat}] median {med:.1f}ms  speedup {BASELINE_MS[cat]/med:.3f}x  "
                  f"accept_len {acc_str}", file=sys.stderr)

    if results:
        medians = [v["median_ms"] for v in results.values()]
        sp_avg = sum(v["speedup_vs_baseline"] for v in results.values()) / len(results)
        results["overall"] = {
            "avg_median_ms": round(statistics.mean(medians), 2),
            "avg_speedup": round(sp_avg, 3),
        }
        print(f"\noverall avg_speedup: {sp_avg:.3f}x  "
              "(realistic — diverse prompts, no prefix-cache repeats)",
              file=sys.stderr)

    output = json.dumps(results, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
