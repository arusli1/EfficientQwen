#!/usr/bin/env python3
"""eval_smoke — 5-prompt-per-task quality probe for fast iteration (~3-5 min).

The lightest pillar in the eval stack:
  eval_smoke.py  (this file)   — N prompts/task (default 5), ~3-5 min, sanity check
  eval_fast.py                 — 10-50% sample,                ~20-25 min, screen
  eval_full.py                 — 100/100/50%,                  ~60 min, cloud parity

Same plumbing as eval_fast.py (shares `eval_common.run_task`), same lm-eval
tasks (mmlu_pro 5-shot no-think, ifeval 0-shot no-think, gpqa_diamond_cot_zeroshot
0-shot thinking-ON), same chat-completions endpoint. Output JSON is
shape-compatible with eval_fast/eval_full (mode="smoke") so existing tooling
that reads results/* keeps working.

What's NOT here:
  - Latency probe (use scripts/bench_latency.py or `make eval-latency`).
  - Pass/fail gating — smoke samples are too small to gate on. Output JSON
    still reports threshold + score so you can eyeball; `all_gates_passed` is
    informational only.

Usage:
  scripts/eval_smoke.py --model-name v6 --container-url http://localhost:8080
  scripts/eval_smoke.py --model-name v6 --limit 10        # 10 prompts/task
  scripts/eval_smoke.py --model-name v6 --out results/v6_smoke.json
"""
from __future__ import annotations

import argparse
import sys
import time

from eval_common import (
    EvalRun, QUALITY_TASKS, print_verdict, run_task,
    utc_stamp, wait_for_ping, write_result,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", required=True)
    p.add_argument("--container-url", default="http://localhost:8080")
    p.add_argument("--out", default=None)
    # Server is expected warm — short ping budget by default.
    p.add_argument("--ping-timeout", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=8)
    # Absolute count per task (lm-eval's `simple_evaluate(limit=N)` with int N
    # = first-N docs per task, deterministic given random_seed). For MMLU-Pro
    # the cap applies per sub-task → 5 means 5*14 ≈ 70 questions total.
    p.add_argument("--limit", type=int, default=5,
                   help="Prompts per task (per sub-task for MMLU-Pro). Default 5.")
    # Smoke runs are short — generous per-task budget mainly catches hangs.
    p.add_argument("--task-timeout", type=float, default=300,
                   help="Per-task wall budget in seconds (0 disables).")
    args = p.parse_args()

    limits = {task[0]: args.limit for task in QUALITY_TASKS}
    out_path = args.out or (
        f"results/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
        f"_{args.model_name}_smoke.json"
    )

    print(f"=== eval_smoke | model={args.model_name} | url={args.container_url} ===")
    print(f"limit: {args.limit} prompts/task (per sub-task for MMLU-Pro)")
    print(f"out: {out_path}")

    if not wait_for_ping(args.container_url, args.ping_timeout):
        print(f"[fatal] {args.container_url}/ping never returned 200", flush=True)
        return 2

    started = utc_stamp()
    t0 = time.perf_counter()
    timeout_s = args.task_timeout if args.task_timeout > 0 else None
    quality = []
    for task in QUALITY_TASKS:
        task_name = task[0]
        quality.append(run_task(args.container_url, task,
                                limit=limits[task_name],
                                concurrency=args.concurrency,
                                timeout_s=timeout_s))

    wall = time.perf_counter() - t0
    # `all_gates_passed` retained for schema compatibility; meaningless at
    # smoke sample sizes — caller should look at scores, not the boolean.
    all_pass = all(t.passed for t in quality)
    run = EvalRun(
        mode="smoke",
        model_name=args.model_name,
        container_url=args.container_url,
        started_utc=started,
        wall_total_s=wall,
        quality=quality,
        latency=None,
        all_gates_passed=all_pass,
        notes={"limit_per_task": args.limit, "concurrency": args.concurrency,
               "task_timeout_s": timeout_s,
               "WARNING": ("smoke sample — gating boolean is informational only; "
                           "promote to eval_fast/eval_full before keeping")},
    )
    write_result(run, out_path)
    print_verdict(run)
    # Smoke exit is always 0 when we actually ran (no fatal infra error) —
    # we don't want CI/loops mis-treating low-n noise as a regression signal.
    return 0


if __name__ == "__main__":
    sys.exit(main())
