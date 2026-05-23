#!/usr/bin/env python3
"""profile_model.py — fast model performance fingerprint for cycle iteration.

The third pillar alongside eval_fast.py (smoke quality) and eval_full.py
(benchmark quality). Captures latency curves + speculative-decoding
acceptance + KV-cache utilization + roofline-validation signals in ~5 min.

Designed by subagent G this session. See docs/PROFILER_DESIGN.md for the
full design rationale and v2/v3 extension plans.

Usage:
  scripts/profile_model.py --model-name cyankiwi-seq8 --container-url http://localhost:8080
  scripts/profile_model.py --model-name v6 --runs 5 --out results/v6_profile.json

Output: a single JSON with prompt-length sweep, MTP acceptance by category,
KV-cache stats, decode/prefill share, and a verdict on which optimization
lever has highest EV.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Reuse eval_common's HTTP plumbing + calib loader + MTP counters
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import (  # noqa: E402
    _http_invoke,
    _invoke_completion,
    _load_calib_prompts,
    _shape,
    _fetch_mtp_counters,
    wait_for_ping,
    BASELINE_LATENCY_MS,
    PROMPT_CONFIGS,
    FILLER,
)

# Default prompt lengths probe the roofline: short → prefill-light /
# decode-bound; long → prefill-amortized / decode-share approaches 1.0.
DEFAULT_PROMPT_LENS = [64, 256, 1024, 4096, 8192]

# Output token target per probe — kept fixed so tokens/sec is comparable
# across prompt lengths.
PROBE_OUTPUT_TOKENS = 128


@dataclass
class PromptLenStat:
    n_prompt_tokens: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    tokens_out_median: int
    tps: float  # tokens/sec
    n_runs: int


@dataclass
class CategoryMtp:
    accept_length: float | None
    drafts: int
    accepted: int


@dataclass
class ProfileResult:
    schema_version: int = 1
    mode: str = "v1-lightweight"
    model_name: str = ""
    container_url: str = ""
    started_utc: str = ""
    wall_total_s: float = 0.0
    cold_start_s: float | None = None
    prompt_len_sweep: dict[int, PromptLenStat] = field(default_factory=dict)
    slope_prefill_ms_per_1k: float | None = None
    slope_decode_intercept_ms: float | None = None
    mtp_by_category: dict[str, CategoryMtp] = field(default_factory=dict)
    # mtp_accepted_per_pos + mtp_k_drop_recommendation are v2 features
    # (requires label-aware Prometheus parsing). Omitted from v1 to avoid
    # advertising always-empty fields. See docs/PROFILER_DESIGN.md.
    kv_cache_usage_peak: float | None = None
    decode_share_overall: float | None = None
    iteration_tokens_per_sec: float | None = None
    inter_token_latency_p50_ms: float | None = None
    ttft_p50_ms: float | None = None
    preemptions_total: int | None = None
    gpu_mem_used_mb: int | None = None
    verdict: dict = field(default_factory=dict)
    git_sha: str = "unknown"
    host: str = ""


def _git_sha() -> str:
    """Repo-root-anchored git HEAD sha (8 chars). 'unknown' on any error."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(repo_root, ".git/HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            with open(os.path.join(repo_root, ".git", head[5:])) as f:
                return f.read().strip()[:8]
        return head[:8]
    except Exception:
        return "unknown"


def _gpu_mem_used_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _scrape_metrics(url: str) -> dict[str, float]:
    """Parse vLLM's Prometheus /metrics endpoint. Returns {metric_name: value}.
    For counters with labels, sums across labels (so accepts e.g. all engines)."""
    try:
        body = urllib.request.urlopen(f"{url}/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Format: 'metric_name{label="x"} 42.0'
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.+eE-]+)', line)
        if not m:
            continue
        name, _labels, value = m.groups()
        try:
            v = float(value)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
    return out


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    k = max(0, min(len(sorted_xs) - 1, int(round(p * (len(sorted_xs) - 1)))))
    return sorted_xs[k]


def prompt_len_sweep(url: str, lens: list[int], runs: int, warmup: int,
                     calib: list[str] | None, seed: int = 0) -> dict[int, PromptLenStat]:
    """For each prompt length n, build a prompt of ~n tokens (real calib if
    available, FILLER otherwise) and time `runs` decodes of PROBE_OUTPUT_TOKENS.
    Returns per-length latency stats."""
    import random
    rng = random.Random(seed)
    out: dict[int, PromptLenStat] = {}
    for n in lens:
        prompts: list[str] = []
        if calib:
            for _ in range(warmup + runs):
                prompts.append(_shape(rng.choice(calib), n))
        else:
            base = FILLER * max(1, n // 10)
            prompts = [base] * (warmup + runs)

        # Warmup (untimed; lets prefix-cache + cudagraph warm to steady-state)
        for p in prompts[:warmup]:
            try:
                _ = _invoke_completion(url, p, PROBE_OUTPUT_TOKENS)
            except Exception:
                pass

        ms: list[float] = []
        toks: list[int] = []
        for p in prompts[warmup:]:
            t0 = time.perf_counter()
            try:
                text, used = _invoke_completion(url, p, PROBE_OUTPUT_TOKENS)
            except Exception as e:
                print(f"  [sweep n={n}] err: {e}", flush=True)
                continue
            ms.append((time.perf_counter() - t0) * 1000)
            toks.append(used)
        if not ms:
            continue
        out[n] = PromptLenStat(
            n_prompt_tokens=n,
            p50_ms=round(_percentile(ms, 0.50), 2),
            p95_ms=round(_percentile(ms, 0.95), 2),
            min_ms=round(min(ms), 2),
            max_ms=round(max(ms), 2),
            tokens_out_median=int(_percentile(toks, 0.5)),
            tps=round(sum(toks) / sum(m / 1000 for m in ms), 2) if sum(ms) > 0 else 0.0,
            n_runs=len(ms),
        )
        print(f"  [n={n:>5}] p50={out[n].p50_ms:>7.1f}ms  p95={out[n].p95_ms:>7.1f}ms  "
              f"tps={out[n].tps:>6.1f}", flush=True)
    return out


def derive_slope(sweep: dict[int, PromptLenStat]) -> tuple[float, float]:
    """Fit p50 ~ a + b * n_prompt (least-squares). Returns (intercept, slope_per_1k).
    intercept ≈ pure decode wall for PROBE_OUTPUT_TOKENS output tokens.
    slope ≈ prefill ms per 1000 input tokens."""
    if len(sweep) < 2:
        return (0.0, 0.0)
    xs = sorted(sweep.keys())
    ys = [sweep[x].p50_ms for x in xs]
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return (sy / n, 0.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return (round(intercept, 2), round(slope * 1000, 2))


def mtp_by_category(url: str, calib: list[str] | None) -> dict[str, CategoryMtp]:
    """For each PROMPT_CONFIGS category (short/medium/long), capture the MTP
    acceptance length delta around one decode."""
    out: dict[str, CategoryMtp] = {}
    for cat, cfg in PROMPT_CONFIGS.items():
        if calib:
            import random
            prompt = _shape(random.Random(0).choice(calib), cfg["num_tokens"])
        else:
            prompt = FILLER * max(1, cfg["num_tokens"] // 10)
        before = _fetch_mtp_counters(url)
        try:
            _ = _invoke_completion(url, prompt, cfg["max_new_tokens"])
        except Exception as e:
            print(f"  [mtp {cat}] err: {e}", flush=True)
            continue
        after = _fetch_mtp_counters(url)
        if before is None or after is None:
            out[cat] = CategoryMtp(accept_length=None, drafts=0, accepted=0)
            continue
        d_accepted = after[0] - before[0]
        d_drafts = after[1] - before[1]
        accept_length = round(1 + d_accepted / d_drafts, 3) if d_drafts > 0 else None
        out[cat] = CategoryMtp(accept_length=accept_length,
                               drafts=d_drafts, accepted=d_accepted)
        print(f"  [mtp {cat:<6}] accept_length={accept_length}", flush=True)
    return out


def k_drop_recommendation(per_pos: list[float], threshold: float = 0.20) -> int | None:
    """v2: largest K where per-position acceptance rate at index K-1 >= threshold.
    Not wired in v1 (requires label-aware metric parsing — see PROFILER_DESIGN.md)."""
    if not per_pos:
        return None
    k = 0
    for i, p in enumerate(per_pos):
        if p >= threshold:
            k = i + 1
        else:
            break
    return k or None


def make_verdict(prof: ProfileResult) -> dict:
    """Per-field decision mapping (mirrors docs/PROFILER_DESIGN.md §"Decision mapping").
    Returns {decode_bound: bool, next_lever: str, rationale: str}.
    None-safe: any field that's None is treated as missing and skipped."""
    ds = prof.decode_share_overall  # may be None on cold container
    decode_bound = (ds is not None) and ds >= 0.55
    long_cat = prof.mtp_by_category.get("long")
    long_accept = long_cat.accept_length if long_cat else None
    reasons = []
    if ds is None:
        reasons.append("decode_share=N/A (no completed e2e samples in window)")
    elif decode_bound:
        reasons.append(f"decode_share={ds:.2f} (≥0.55) → decode-bound")
    else:
        reasons.append(f"decode_share={ds:.2f} (<0.55) → STOP pursuing lm_head; switch axis")
    if long_accept is not None and 3.0 < long_accept < 4.5:
        reasons.append(f"mtp.long.accept_length={long_accept} (3-4.5) → MTP redistill high-EV (~1.2×)")
    # k_drop_recommendation is a v2 feature (label-aware metrics parse); skip.
    if (prof.kv_cache_usage_peak or 0) > 0.80:
        reasons.append("kv_cache.peak > 0.80 → KV pressure; don't bump max_num_seqs")
    next_lever = "L_lmhead_quant" if decode_bound else (
        "L_prefill_or_scheduler" if ds is not None else "UNKNOWN_INSUFFICIENT_DATA"
    )
    return {
        "decode_bound": decode_bound,
        "next_lever": next_lever,
        "rationale": " · ".join(reasons),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", required=True)
    p.add_argument("--container-url", default="http://localhost:8080")
    p.add_argument("--out", default=None)
    p.add_argument("--ping-timeout", type=int, default=60,
                   help="Short by default — assume server is already warm.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--prompt-lens", default=",".join(map(str, DEFAULT_PROMPT_LENS)),
                   help=f"Comma-separated. Default: {DEFAULT_PROMPT_LENS}")
    p.add_argument("--torch-profiler", action="store_true",
                   help="Run with torch.profiler integration to capture per-op "
                        "CUDA timings (to verify the lm_head decode-share "
                        "assumption directly). GPU-only; emits "
                        "results/profile/<v>/op_share_<cat>.json alongside the "
                        "standard fingerprint. The lightweight default profiler "
                        "infers decode share rather than measuring per-op timings; "
                        "this is the higher-fidelity path. Not yet implemented.")
    args = p.parse_args()

    if args.torch_profiler:
        print("[profile] --torch-profiler requested.", flush=True)
        print("[profile] Implementation plan (GPU-side):", flush=True)
        print("[profile]   1. Wrap each prompt-length sweep in torch.profiler.profile()", flush=True)
        print("[profile]      with activities=[CPU, CUDA] and "
              "with_stack=False.", flush=True)
        print("[profile]   2. After each cat completes, export key_averages() "
              "filtered to:", flush=True)
        print("[profile]      - gemv2T_kernel_val<__half> (cuBLAS FP16 lm_head GEMV)", flush=True)
        print("[profile]      - awq_marlin GEMM kernels (W4 MLP+attn projections)", flush=True)
        print("[profile]      - aten::sampler / random_sample / topk_topp_sampler", flush=True)
        print("[profile]      - flash_attn_with_kvcache, mamba2_*, gdn_*", flush=True)
        print("[profile]   3. Compute per-op self_ms, percent_of_decode, call_count.", flush=True)
        print("[profile]   4. Emit results/profile/<v>/op_share_<cat>.json with "
              "schema_version='1.0', date, measurement_type='op_share', verdict, "
              "and per-cat per-op breakdown.", flush=True)
        print("[profile]   5. Validate via scripts/check_schemas.py (Schema B).", flush=True)
        print("[profile] Stub flag — full implementation pending. Falling back "
              "to v1-lightweight profile.", flush=True)
        # Continue with v1 path; flag is informational only for now

    lens = [int(x) for x in args.prompt_lens.split(",") if x.strip()]
    out_path = args.out or f"results/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}_{args.model_name}_profile.json"

    print(f"=== profile_model v1 | model={args.model_name} | url={args.container_url} ===", flush=True)
    print(f"  prompt_lens: {lens} | runs: {args.runs} | out: {out_path}", flush=True)

    t_ping_start = time.perf_counter()
    if not wait_for_ping(args.container_url, args.ping_timeout):
        print(f"[fatal] /ping never returned 200", flush=True)
        return 2
    cold_start_s = round(time.perf_counter() - t_ping_start, 1)

    prof = ProfileResult(
        model_name=args.model_name,
        container_url=args.container_url,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cold_start_s=cold_start_s,
        git_sha=_git_sha(),
        host=socket.gethostname(),
    )

    t0 = time.perf_counter()
    calib = _load_calib_prompts()
    if not calib:
        print("[warn] calib_1024.jsonl missing — using FILLER (inflated speedup)", flush=True)

    print("\n[1/4] Scraping baseline /metrics...", flush=True)
    metrics_before = _scrape_metrics(args.container_url)

    print("\n[2/4] Prompt-length sweep...", flush=True)
    prof.prompt_len_sweep = prompt_len_sweep(
        args.container_url, lens, args.runs, args.warmup, calib,
    )
    prof.slope_decode_intercept_ms, prof.slope_prefill_ms_per_1k = derive_slope(prof.prompt_len_sweep)
    print(f"  slope: prefill={prof.slope_prefill_ms_per_1k}ms/1k_prompt  "
          f"decode_intercept={prof.slope_decode_intercept_ms}ms", flush=True)

    print("\n[3/4] MTP acceptance by category...", flush=True)
    prof.mtp_by_category = mtp_by_category(args.container_url, calib)

    print("\n[4/4] Final /metrics scrape + derivations...", flush=True)
    metrics_after = _scrape_metrics(args.container_url)
    # KV cache: gauge, use after-value directly
    prof.kv_cache_usage_peak = round(
        metrics_after.get("vllm:kv_cache_usage_perc", 0.0), 4
    )
    # Histograms expose `_sum` (cumulative seconds) and `_count` (samples).
    # We use `_sum` here for "total time in this state across the window".
    decode_delta = (metrics_after.get("vllm:request_decode_time_seconds_sum", 0.0)
                    - metrics_before.get("vllm:request_decode_time_seconds_sum", 0.0))
    e2e_delta = (metrics_after.get("vllm:e2e_request_latency_seconds_sum", 0.0)
                 - metrics_before.get("vllm:e2e_request_latency_seconds_sum", 0.0))
    prof.decode_share_overall = round(decode_delta / e2e_delta, 4) if e2e_delta > 0 else None
    # iteration_tokens_total is a Histogram → use `_sum` for total tokens
    # processed across all iterations in the window.
    iter_delta = (metrics_after.get("vllm:iteration_tokens_total_sum", 0.0)
                  - metrics_before.get("vllm:iteration_tokens_total_sum", 0.0))
    prof.iteration_tokens_per_sec = round(iter_delta / max(1.0, e2e_delta), 2) if iter_delta > 0 else None
    # Counter exposed as `_total` per Prometheus convention.
    prof.preemptions_total = int(metrics_after.get("vllm:num_preemptions_total", 0.0))
    prof.gpu_mem_used_mb = _gpu_mem_used_mb()
    prof.verdict = make_verdict(prof)

    prof.wall_total_s = round(time.perf_counter() - t0, 1)

    # Atomic write — mirrors eval_common.write_result pattern
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = dataclasses.asdict(prof)
    payload["prompt_len_sweep"] = {str(k): dataclasses.asdict(v)
                                   for k, v in prof.prompt_len_sweep.items()}
    payload["mtp_by_category"] = {k: dataclasses.asdict(v)
                                  for k, v in prof.mtp_by_category.items()}
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)

    print(f"\n[done] wrote {out_path}  ({prof.wall_total_s:.1f}s wall)", flush=True)
    print(f"  decode_share={prof.decode_share_overall}  kv_peak={prof.kv_cache_usage_peak}  "
          f"iter_tps={prof.iteration_tokens_per_sec}")
    print(f"  verdict: {prof.verdict.get('next_lever')}")
    print(f"  rationale: {prof.verdict.get('rationale')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
