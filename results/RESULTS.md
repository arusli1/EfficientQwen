# Results

Benchmark records for the EfficientQwen serving stack. Hardware: AWS g5.xlarge
(1× NVIDIA A10G, 24 GB). Baseline = BF16 Qwen3.5-4B under stock vLLM.

## Headline

| Config | Speedup vs BF16 | MMLU-Pro (≥0.621) | IFEval (≥0.814) | GPQA-D |
|---|---:|---:|---:|---:|
| `cyankiwi-seq8` (AWQ-4bit + MTP + batching) | **3.45×** | 0.650 ✓ | 0.833 ✓ | 0.586 |

MMLU-Pro and IFEval clear their floors with margin. GPQA-Diamond is the model's
weakest task — the local full-pool mean is ≈ 0.64, but thinking-mode traces
frequently hit the generation length cap, which suppresses the scored mean. It
is reported, not tuned for.

## Speculative-decoding depth sweep

`mtp_k_sweep.json` — MTP candidate depth swept K=2..7. Acceptance length rises
monotonically with K, but total speedup peaks at **K=4** (inverted-U): past 4
candidates the extra draft work outweighs the higher acceptance on long-form
generation.

| K | short | medium | avg | accept (short/med) |
|---|---:|---:|---:|---:|
| 2 | 3.55 | 2.89 | 3.22 | 2.68 / 2.54 |
| 3 | 3.62 | 2.97 | 3.29 | 3.26 / 3.12 |
| **4** | **3.95** | 2.84 | **3.40** | 3.75 / 3.43 |
| 5 | 3.96 | 2.71 | 3.33 | 4.21 / 3.76 |
| 6 | 3.97 | 2.57 | 3.27 | 4.45 / 3.97 |
| 7 | 3.77 | 2.48 | 3.13 | 4.57 / 4.16 |

K=4 is +8.6% over the K=7 default on the short+medium mix → it is the depth used
by the `cyankiwi-seq8-mtp4` variant.

## Bottleneck profile

`profile/cyankiwi-seq8-clean/` — per-category serving profile. Decode share is
0.60–0.74 across short/medium/long prompts, i.e. the workload is decode-bound,
which is what makes weight quantization and speculative decoding the high-leverage
levers (rather than prefill/attention optimizations).
