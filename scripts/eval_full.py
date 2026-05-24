#!/usr/bin/env python3
"""eval_full — full-sample lm-eval-harness gate for any served model.

Use as the ground-truth check before tagging a result KEEP. Drives lm-eval
against the running vLLM via OpenAI chat-completions: MMLU-Pro 100% /
IFEval 100% / GPQA-Diamond 50% (n=98) by default.

Latency: 5 runs × 3 categories using DIVERSE prompts from
data/calibration_1024.jsonl when present. Falls back to FILLER
with a clear "inflated" flag in the output JSON if calib data is unavailable.

Usage:
  # Server must already be up at --container-url (e.g. host vLLM or docker)
  scripts/eval_full.py --model-name cyankiwi-seq8 --container-url http://localhost:8080
  scripts/eval_full.py --model-name v6 --out results/v6_full.json --skip-latency
  scripts/eval_full.py --model-name probe --mmlu-pro 1.0 --ifeval 1.0 --gpqa 0.5

Exit code: 0 if all gates pass, 1 otherwise. Result JSON always written.
"""
from __future__ import annotations

import argparse
import sys
import time

from eval_common import (
    EvalRun, QUALITY_TASKS, print_verdict, run_latency, run_task,
    utc_stamp, wait_for_ping, write_result,
)


# Defaults: full sample for MMLU-Pro and IFEval, 50% for GPQA (n=99).
FULL_LIMITS = {"mmlu_pro": None, "ifeval": None, "gpqa_diamond_cot_zeroshot": 0.5}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", required=True,
                   help="Label for the result JSON (e.g. 'cyankiwi-seq8', 'v6-w4a16').")
    p.add_argument("--container-url", default="http://localhost:8080")
    p.add_argument("--out", default=None,
                   help="Result JSON path (default: results/{stamp}_{model}_full.json)")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--ping-timeout", type=int, default=900)
    p.add_argument("--concurrency", type=int, default=8)
    # Per-task limit overrides (default values match cloud).
    p.add_argument("--mmlu-pro", type=float, default=None,
                   help="Limit for MMLU-Pro (None=full).")
    p.add_argument("--ifeval", type=float, default=None,
                   help="Limit for IFEval (None=full).")
    p.add_argument("--gpqa", type=float, default=0.5,
                   help="Limit for GPQA-Diamond (default 0.5 → n=99).")
    p.add_argument("--latency-runs", type=int, default=5)
    p.add_argument("--latency-warmup", type=int, default=2)
    p.add_argument("--task-timeout", type=float, default=1500,
                   help="Per-task wall-time budget in seconds. Exceedance "
                        "marks the task TIMEOUT. Pass 0 to disable.")
    # --smoke-first: 10%-sample quick-abort gate. Runs a 10% sample first and
    # aborts (exit 3) if ANY task drops > 5pp under the declared baseline —
    # catches regressions in ~12 min instead of waiting the full 60 min for
    # the failure to materialize at full sample.
    p.add_argument("--smoke-first", action="store_true",
                   help="Run a 10%-sample quick gate first; abort if any task "
                        "drops > --smoke-gate-pp under declared baseline.")
    p.add_argument("--smoke-gate-pp", type=float, default=5.0,
                   help="Max pp drop vs baseline before quick-gate aborts (default 5).")
    p.add_argument("--baseline-mmlu-pro", type=float, default=None,
                   help="Declared MMLU-Pro baseline (for --smoke-first gate).")
    p.add_argument("--baseline-ifeval", type=float, default=None,
                   help="Declared IFEval baseline (for --smoke-first gate).")
    p.add_argument("--baseline-gpqa", type=float, default=None,
                   help="Declared GPQA-D baseline (for --smoke-first gate).")
    args = p.parse_args()

    # argparse default of None for --mmlu-pro/--ifeval already means "full
    # sample"; --gpqa defaults to 0.5 (the cloud sample). Explicit overrides
    # from the caller win.
    limits = {
        "mmlu_pro": args.mmlu_pro,
        "ifeval": args.ifeval,
        "gpqa_diamond_cot_zeroshot": args.gpqa,
    }

    out_path = args.out or f"results/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}_{args.model_name}_full.json"

    print(f"=== eval_full | model={args.model_name} | url={args.container_url} ===")
    print(f"limits: {limits}")
    print(f"out: {out_path}")

    if not wait_for_ping(args.container_url, args.ping_timeout):
        print(f"[fatal] {args.container_url}/ping never returned 200", flush=True)
        return 2

    started = utc_stamp()
    t0 = time.perf_counter()
    timeout_s = args.task_timeout if args.task_timeout > 0 else None

    # --smoke-first quick-abort gate: run a tiny sample first and bail early
    # if any task regresses badly vs the declared baseline, before spending the
    # full eval budget.
    if args.smoke_first:
        baselines = {
            "mmlu_pro": args.baseline_mmlu_pro,
            "ifeval": args.baseline_ifeval,
            "gpqa_diamond_cot_zeroshot": args.baseline_gpqa,
        }
        if all(v is None for v in baselines.values()):
            print("[smoke-first] WARNING no --baseline-* declared; running gate in "
                  "informational mode only (will not abort)", flush=True)
        print("[smoke-first] running 10%-sample quick gate first "
              "(abort threshold: > {:.1f} pp under baseline)"
              .format(args.smoke_gate_pp), flush=True)
        smoke_limits = {k: 0.1 for k in baselines}
        smoke_quality = []
        smoke_t0 = time.perf_counter()
        for task in QUALITY_TASKS:
            task_name = task[0]
            smoke_quality.append(run_task(args.container_url, task,
                                          limit=smoke_limits[task_name],
                                          concurrency=args.concurrency,
                                          timeout_s=timeout_s))
        smoke_wall = time.perf_counter() - smoke_t0
        # Check each task against its declared baseline
        aborted = []
        for sq in smoke_quality:
            baseline = baselines.get(sq.task)
            if baseline is None:
                continue
            drop_pp = (baseline - sq.score) * 100
            if drop_pp > args.smoke_gate_pp:
                aborted.append((sq.task, sq.score, baseline, drop_pp))
        if aborted:
            print(f"\n[smoke-first] ABORT — {len(aborted)} task(s) failed quick gate "
                  f"(took {smoke_wall:.0f}s):", flush=True)
            for task, score, baseline, drop_pp in aborted:
                print(f"  {task}: {score:.4f} vs baseline {baseline:.4f} "
                      f"(-{drop_pp:.1f} pp > {args.smoke_gate_pp} pp threshold)",
                      flush=True)
            print(f"[smoke-first] full eval skipped; results aborted-with-smoke.",
                  flush=True)
            wall = time.perf_counter() - t0
            run = EvalRun(
                mode="full",
                model_name=args.model_name,
                container_url=args.container_url,
                started_utc=started,
                wall_total_s=wall,
                quality=smoke_quality,
                latency=None,
                all_gates_passed=False,
                notes={"aborted_smoke_first": True,
                       "smoke_gate_pp": args.smoke_gate_pp,
                       "smoke_limits": smoke_limits,
                       "smoke_wall_s": smoke_wall,
                       "aborted_tasks": [
                           {"task": t, "score": s, "baseline": b, "drop_pp": d}
                           for t, s, b, d in aborted
                       ]},
            )
            write_result(run, out_path)
            print_verdict(run)
            return 3  # distinct exit for "quick-gate abort"
        print(f"[smoke-first] PASS — quick gate cleared in {smoke_wall:.0f}s; "
              "proceeding to full eval", flush=True)

    quality = []
    for task in QUALITY_TASKS:
        task_name = task[0]
        quality.append(run_task(args.container_url, task,
                                limit=limits[task_name],
                                concurrency=args.concurrency,
                                timeout_s=timeout_s))

    latency = None
    if not args.skip_latency:
        latency = run_latency(args.container_url,
                              runs=args.latency_runs,
                              warmup=args.latency_warmup)

    wall = time.perf_counter() - t0
    all_pass = all(t.passed for t in quality)
    run = EvalRun(
        mode="full",
        model_name=args.model_name,
        container_url=args.container_url,
        started_utc=started,
        wall_total_s=wall,
        quality=quality,
        latency=latency,
        all_gates_passed=all_pass,
        notes={"limits": {k: limits[k] for k in limits},
               "concurrency": args.concurrency,
               "task_timeout_s": timeout_s},
    )
    write_result(run, out_path)
    print_verdict(run)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
