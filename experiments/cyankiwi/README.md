# cyankiwi — AWQ-4bit reference

The quality reference for the project: Qwen3.5-4B quantized to 4-bit and served
with MTP speculative decoding, single stream. Every other variant is a delta on
top of this and is measured against it.

## Identity

- **Weights**: `weights/cyankiwi/` — Qwen3.5-4B, AWQ-calibrated to 4-bit
  (compressed-tensors W4A16, `group_size=32`, symmetric) on the MLP `Linear`
  layers. `lm_head` and embeddings stay FP16.
- **Architecture**: Qwen3.5 hybrid, 32 layers (24 Mamba2/GDN + 8 full-attention
  at indices 3, 7, 11, 15, 19, 23, 27, 31). Hidden size 2560, vocab 248320.
- **Speculative decoding**: MTP head, K=7 candidates per step.
- **Serving**: vLLM, `max_num_seqs=1`. See `config.env` for the full flag set.

## Notes

- `lm_head` is left in FP16 deliberately — it dominates decode memory traffic,
  and 4-bit-quantizing it costs more quality than the speed is worth at this size.
- Cold-start is reduced from ~697 s to ~156 s by pre-baking the `torch.compile`
  cache and matching the inductor cache key across the build and serve GPUs
  (`scripts/_cache_patch.py`, `scripts/bake_cache.py`).

## Reproduce

```bash
make download                            # -> weights/cyankiwi/
make serve VARIANT=cyankiwi              # uses experiments/cyankiwi/config.env
# in another terminal:
make eval-quality-full VARIANT=cyankiwi
make eval-latency      VARIANT=cyankiwi
```
