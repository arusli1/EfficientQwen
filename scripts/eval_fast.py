#!/usr/bin/env python3
"""eval_fast — ~20-25 min smoke eval for cycle iteration. Trades some MMLU-Pro
and IFEval precision for speed; keeps GPQA at 50% (the cloud sample) because
GPQA is the noisiest gate and the one most likely to regress.

Use between cycles to screen candidates fast. Pass eval_fast → promote to
scripts/eval_full.py for the definitive cloud-matching score before submission.

Default sample sizes (vs eval_full):
  MMLU-Pro: 10%  (~120 per subject × 14 subjects ≈ 1200 of 12032)  — full=100%
  IFEval:   30%  (~162 of 541 instructions)                         — full=100%
  GPQA-D:   50%  (~99 of 198 = cloud's exact deterministic subset)  — same as full (cloud)

Sampling notes:
- MMLU-Pro is exposed by lm-eval as 14 sub-tasks (mmlu_pro_math,
  mmlu_pro_law, ...). `limit=0.10` applies PER sub-task → naturally
  proportional-stratified across all 14 subjects (not the first-1200
  biased toward early subjects). Confirmed safe by reading
  `lm_eval/tasks/mmlu_pro/` structure.
- IFEval has no subtask grouping; `limit=0.30` is first-162 in dataset
  order. lm-eval shuffles via `random_seed` but order within shuffle is
  deterministic given the seed → reproducible smoke.
- GPQA at 0.50 (first-99 deterministic) MATCHES cloud's exact sample,
  so eval_fast.gpqa ≈ cloud.gpqa modulo vLLM nondeterminism (±2-5pp).

GPQA at 50% dominates wall time (~15 min with thinking-on, max 12288 tokens
per question). Cutting GPQA further makes noise > signal — Slack guidance
from Alireza: "I can get 50→70% even with the base model" at n=98.

Usage identical to eval_full.py:
  scripts/eval_fast.py --model-name v6 --container-url http://localhost:8080
  scripts/eval_fast.py --model-name v6 --skip-latency  # ~20 min total
"""
from __future__ import annotations

import argparse
import sys
import time

from eval_common import (
    EvalRun, QUALITY_TASKS, print_verdict, run_latency, run_task,
    utc_stamp, wait_for_ping, write_result,
)


SMOKE_LIMITS = {"mmlu_pro": 0.10, "ifeval": 0.30,
                "gpqa_diamond_cot_zeroshot": 0.50}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", required=True)
    p.add_argument("--container-url", default="http://localhost:8080")
    p.add_argument("--out", default=None)
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--ping-timeout", type=int, default=900)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--mmlu-pro", type=float, default=SMOKE_LIMITS["mmlu_pro"])
    p.add_argument("--ifeval", type=float, default=SMOKE_LIMITS["ifeval"])
    p.add_argument("--gpqa", type=float,
                   default=SMOKE_LIMITS["gpqa_diamond_cot_zeroshot"])
    p.add_argument("--latency-runs", type=int, default=3)
    p.add_argument("--latency-warmup", type=int, default=2)
    # Smoke timeout scales with smoke sample: MMLU-Pro at 10% should take
    # roughly 10% of full's ~25 min budget plus the lm-eval setup overhead.
    # 600s (10 min) is conservative; user can override.
    p.add_argument("--task-timeout", type=float, default=600,
                   help="Per-task wall budget in seconds (0 disables).")
    args = p.parse_args()

    limits = {
        "mmlu_pro": args.mmlu_pro,
        "ifeval": args.ifeval,
        "gpqa_diamond_cot_zeroshot": args.gpqa,
    }
    out_path = args.out or f"results/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}_{args.model_name}_fast.json"

    print(f"=== eval_fast | model={args.model_name} | url={args.container_url} ===")
    print(f"limits: {limits}")
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

    latency = None
    if not args.skip_latency:
        latency = run_latency(args.container_url,
                              runs=args.latency_runs,
                              warmup=args.latency_warmup)

    wall = time.perf_counter() - t0
    all_pass = all(t.passed for t in quality)
    run = EvalRun(
        mode="fast",
        model_name=args.model_name,
        container_url=args.container_url,
        started_utc=started,
        wall_total_s=wall,
        quality=quality,
        latency=latency,
        all_gates_passed=all_pass,
        notes={"limits": limits, "concurrency": args.concurrency,
               "task_timeout_s": timeout_s,
               "WARNING": "smoke sample — promote to eval_full before keeping"},
    )
    write_result(run, out_path)
    print_verdict(run)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
