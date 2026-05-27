# EfficientQwen

Inference optimization of **Qwen3.5-4B** on a single **NVIDIA A10G (24 GB)**:
4-bit weight quantization, MTP speculative decoding, and a tuned vLLM serving
stack, packaged in a reproducible Docker container.

**Headline: ~3.4× faster than the BF16 baseline at matched quality on MMLU-Pro
and IFEval**, served with a one-line `make serve`.

---

## Optimization stack

```
  Qwen3.5-4B  (BF16, ~4B params)
      │
      ▼
  ┌────────────────────────────────────────────────┐
  │  AWQ 4-bit weights                              │  compressed-tensors W4A16, g=32
  │  ↳ MLP Linear layers; lm_head kept FP16         │  weights/cyankiwi/recipe.yaml
  └────────────────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────────────────┐
  │  MTP speculative decoding  (K=4)                │  multi-token-predictor head
  │  ↳ K=4 selected by a depth sweep                │  results/mtp_k_sweep.json
  └────────────────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────────────────┐
  │  vLLM serving runtime                           │  scripts/serve.py
  │   • max_num_seqs=8, matched CUDA graphs         │
  │   • chunked prefill + prefix caching            │
  │   • qk-norm + RoPE fusion, cuDNN prefill        │
  │   • optional FP8 KV cache (memory variant)      │
  └────────────────────────────────────────────────┘
      │
      ▼
  ┌────────────────────────────────────────────────┐
  │  Pre-baked torch.compile cache                  │  scripts/bake_cache.py
  │  ↳ build/serve GPU device-name shim             │  scripts/_cache_patch.py
  │  ↳ cold start 697s → 156s (~78% off)            │
  └────────────────────────────────────────────────┘
```

| Lever | Technique | Source |
|---|---|---|
| **Weight quant** | AWQ 4-bit (compressed-tensors W4A16, g=32) on MLPs | `weights/cyankiwi/recipe.yaml` |
| **Speculative decoding** | Multi-token-predictor head, K=4 (sweep-selected) | `results/mtp_k_sweep.json` |
| **Batching** | `max_num_seqs=8`, multi-size CUDA graphs | `experiments/cyankiwi-seq8/config.env` |
| **Kernel fusion** | qk-norm + RoPE fusion, cuDNN prefill | `scripts/serve.py` |
| **KV cache** | block size 16, chunked prefill, prefix caching; optional FP8 KV | `scripts/serve.py` |
| **Cold start** | pre-baked `torch.compile` cache + device-name shim | `scripts/bake_cache.py`, `scripts/_cache_patch.py` |

---

## Results

Measured on **AWS g5.xlarge** (1× A10G, 24 GB). Baseline = BF16 Qwen3.5-4B under
stock vLLM. Latency speedup is the average over a mixed short/medium/long prompt
set. Quality floors are the competition thresholds.

| Variant | Speedup | MMLU-Pro (≥0.621) | IFEval (≥0.814) | GPQA-D |
|---|---:|---:|---:|---:|
| `cyankiwi` — AWQ-4bit reference | 1.0× | 0.65 | 0.83 | 0.59 |
| `cyankiwi-seq8` — + batching | **3.45×** | **0.65** ✓ | **0.83** ✓ | 0.59 |
| `cyankiwi-seq8-mtp4` — + K=4 MTP tuning | +8% over seq8¹ | 0.64 ✓ | 0.86 ✓ | — |

MMLU-Pro and IFEval clear their floors with margin. GPQA-Diamond is the model's
weakest task (local full-pool mean ≈ 0.64, with thinking-mode generations often
hitting the length cap, which suppresses the scored mean); it is reported here
rather than tuned for.

¹ `cyankiwi-seq8-mtp4` latency is from local benchmarks (`bench_latency.py`),
relative to `cyankiwi-seq8`; quality is from a 10% eval sample.

---

## Quick start

```bash
make install            # .venv + host deps
make download           # weights/cyankiwi/  (~3.8 GB)
make test               # pytest (~10s, no GPU)
```

### Serve + evaluate (GPU host)

```bash
make serve                                   # default variant
make eval-latency                            # latency probe
make eval-quality-full                       # full lm-eval sample

# pick a variant:
make serve         VARIANT=cyankiwi-seq8-mtp4
make eval-latency  VARIANT=cyankiwi-seq8-mtp4
```

Outputs land in `experiments/<variant>/{quality,latency}_<date>.json`.

### Container build

```bash
make build                       # docker build with native cache bake (GPU host)
make build-import                # build using a pre-built cache_import.tar.gz
make verify-image VARIANT=cyankiwi
```

---

## Repo layout

```
experiments/                one self-contained directory per serving variant
  cyankiwi/                   AWQ-4bit reference (MTP K=7, single stream)
  cyankiwi-seq8/              + max_num_seqs=8        — the ~3.4× measured config
  cyankiwi-seq8-mtp4/         + MTP K=4 + tuned CUDA graphs (latency)
  cyankiwi-seq8-mtp4-fp8kv/   + FP8 KV cache (memory)
  README.md                  variant catalog + naming convention
scripts/                    serving, benchmarking, eval, and quantization tooling
eval/                       lm-eval-harness driver scripts
results/                    benchmark records (K-sweep, latency, profiles)
tests/                      pytest suite mirroring the critical scripts
weights/                    checkpoints (gitignored; `make download`)
Dockerfile  Makefile        VARIANT-aware build + run
```

Each `experiments/<variant>/` holds a `README.md` (what changed and why), a
serve `config.env`, and a `measurements.json` record validated by
`make check-schemas`.

---

## Notes

- **Quantization**: `lm_head` is deliberately kept in FP16 — it dominates decode
  memory traffic, and 4-bit-quantizing it costs more quality than the speed gains
  at this model size.
- **Speculative depth**: MTP initializes from the base model and adds little
  training overhead. A depth sweep selected K=4 — acceptance rate falls past 4
  candidates on long-form generation, so deeper drafting wastes compute.
- **Cold start**: `torch._inductor` keys its compile cache by GPU SM string, so a
  cache baked on one GPU misses on another. `_cache_patch.py` normalizes the
  device name at interpreter startup (including vLLM worker subprocesses, via
  `sitecustomize.py`) so the serve host hits the warm cache.

## License

See [`LICENSE`](LICENSE).
