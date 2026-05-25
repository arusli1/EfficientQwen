# cyankiwi-seq8-mtp4-fp8kv — memory-optimized

[`cyankiwi-seq8-mtp4`](../cyankiwi-seq8-mtp4/) plus an **FP8 (E4M3) KV cache**
(`--kv-cache-dtype fp8`). Quantizing the KV cache to 8-bit roughly halves its
memory footprint, freeing headroom for longer contexts or larger batches on a
24 GB A10G. Because KV quantization only affects attention-cache reads, quality
impact is expected to be negligible.

Provided as a ready-to-serve configuration; it shares the reference weights and
is not separately benchmarked here.

## Config delta vs `cyankiwi-seq8-mtp4`

| Var | value | Effect |
|---|---|---|
| `VLLM_KV_CACHE_DTYPE` | `fp8` | ~2× smaller KV cache (E4M3) |

## Reproduce

```bash
make serve VARIANT=cyankiwi-seq8-mtp4-fp8kv
make eval-latency      VARIANT=cyankiwi-seq8-mtp4-fp8kv
make eval-quality      VARIANT=cyankiwi-seq8-mtp4-fp8kv
```
