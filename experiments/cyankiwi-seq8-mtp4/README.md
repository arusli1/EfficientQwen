# cyankiwi-seq8-mtp4 — latency-optimized

[`cyankiwi-seq8`](../cyankiwi-seq8/) with three runtime tunings stacked on top.
Weights and chat template are unchanged, so quality is expected to match the
reference; the change is purely how the speculative decoder and CUDA graphs are
configured.

## Levers

| Lever | Change | Effect |
|---|---|---|
| **MTP depth** | K=7 → **K=4** | sweep optimum (`results/mtp_k_sweep.json`); fewer wasted draft tokens per step |
| **CUDA-graph capture** | sizes matched to K=4 × seqs∈{1..8} | avoids cold-start recompiles and shape misses |
| **Repetition penalty** | → 1.0 | keeps MTP acceptance high on long generations |

## Measured (local)

- **Latency**: `bench_latency.py` (n=10, warmup=3, cold-mixed) — **+8.1% average**
  decode vs `cyankiwi-seq8` (short +17.0%, medium +8.1%, long +1.2%).
- **Quality (10% sample)**: MMLU-Pro 0.642 (floor 0.621) ✓, IFEval 0.862
  (floor 0.814) ✓ — consistent with the quality-neutral expectation for a
  runtime-only change.

## Reproduce

```bash
make serve VARIANT=cyankiwi-seq8-mtp4
make eval-latency      VARIANT=cyankiwi-seq8-mtp4
make eval-quality      VARIANT=cyankiwi-seq8-mtp4
```
