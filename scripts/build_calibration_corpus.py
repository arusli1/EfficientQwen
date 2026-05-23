#!/usr/bin/env python3
"""Regenerate the calibration corpus from public datasets.

Produces a deterministic, fixed-seed prompt corpus from public data that
downstream tools use as their canonical calibration source:
    - scripts/llmcompressor_calibrate.py  (AWQ weight calibration)
    - scripts/bench_latency.py            (--calib default)

Source mix (deterministic, public, ungated):
  - MMLU-Pro test split (TIGER-Lab/MMLU-Pro, public, cached locally)
  - IFEval train split (wis-k/instruction-following-eval, public, cached)
  No GPQA — gated dataset; requires HF token. Skipping it adds ~1-3pp of
  drift vs the original mix.

Output schema (compatible with bench_latency.py / llmcompressor / dump_*):
  {"text": "...", "source": "mmlu-pro|ifeval", "split": "...", "seed": 42}

Usage:
  python3 scripts/build_calibration_corpus.py \\
      --output data/calibration_1024.jsonl \\
      --n 1024 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _load_mmlu_pro_prompts(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: list[dict] = []
    for i in idxs[:n]:
        row = ds[i]
        q = row.get("question", "").strip()
        opts = row.get("options") or []
        if not q:
            continue
        opts_txt = "\n".join(f"({chr(65 + j)}) {o}" for j, o in enumerate(opts))
        text = f"{q}\n\n{opts_txt}\n\nAnswer with the letter only."
        if len(text) >= 50:
            out.append({"text": text, "source": "mmlu-pro", "split": "test"})
    return out


def _load_ifeval_prompts(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("wis-k/instruction-following-eval", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: list[dict] = []
    for i in idxs[:n]:
        row = ds[i]
        text = (row.get("prompt") or row.get("instruction") or "").strip()
        if text and len(text) >= 30:
            out.append({"text": text, "source": "ifeval", "split": "train"})
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=1024, help="target prompt count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mix-mmlu", type=float, default=0.6,
        help="fraction of prompts from MMLU-Pro (rest from IFEval). 0.6 ≈ "
             "matches the original distribution")
    args = p.parse_args()

    rng = random.Random(args.seed)
    n_mmlu = int(args.n * args.mix_mmlu)
    n_ifeval = args.n - n_mmlu

    print(f"sampling {n_mmlu} from mmlu-pro + {n_ifeval} from ifeval (seed={args.seed})",
          file=sys.stderr)
    mmlu = _load_mmlu_pro_prompts(n_mmlu * 2, rng)[:n_mmlu]
    ifeval = _load_ifeval_prompts(n_ifeval * 2, rng)[:n_ifeval]

    prompts = mmlu + ifeval
    rng.shuffle(prompts)
    for row in prompts:
        row["seed"] = args.seed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in prompts:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
