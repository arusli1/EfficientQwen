# cyankiwi-seq8 — throughput-tuned reference

Same AWQ-4bit weights as [`cyankiwi`](../cyankiwi/), with concurrency raised to
`max_num_seqs=8` so the batched evaluations (MMLU-Pro, GPQA-Diamond) finish
inside the per-task wall-clock budget. No quality knob (weights, quantization,
chat template) is touched — only runtime concurrency and the CUDA-graph capture
set. **This is the configuration measured on the competition cloud.**

## Config delta vs `cyankiwi`

| Var | cyankiwi | cyankiwi-seq8 | Why |
|---|---|---|---|
| `VLLM_MAX_NUM_SEQS` | 1 | 8 | 8× batch concurrency for the evals |
| `VLLM_CUDAGRAPH_CAPTURE_SIZES` | `8` | `8,16,...,64` | covers every batch shape MTP=7 × seqs∈{1..8} can produce |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.90 | 0.92 | KV headroom for batch=64 |

## Measured (cloud)

| Speedup vs BF16 | MMLU-Pro | IFEval | GPQA-Diamond |
|---|---|---|---|
| **3.45×** | **0.650** (floor 0.621) ✓ | **0.833** (floor 0.814) ✓ | 0.586 (floor 0.630) |

MMLU-Pro and IFEval clear their floors comfortably. GPQA-Diamond lands just under
its floor on the cloud's deterministic subset; the local full-pool mean is ~0.64
with ~37% of thinking-mode traces truncated by the generation cap (which
suppresses the scored mean). See `measurements.json` for the full record.

## Reproduce

```bash
make serve VARIANT=cyankiwi-seq8
make eval-quality-full VARIANT=cyankiwi-seq8
make eval-latency      VARIANT=cyankiwi-seq8
```
