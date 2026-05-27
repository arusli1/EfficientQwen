# Variant catalog

One subdirectory per serving variant. Each is self-contained — a serve
`config.env`, a `README.md` describing the change, and a `measurements.json`
record validated by `scripts/check_schemas.py` (`make check-schemas`). All
variants below share the same `weights/cyankiwi/` checkpoint and differ only in
runtime configuration.

| Variant | Optimization | What it's for |
|---|---|---|
| [`cyankiwi`](cyankiwi/) | AWQ-4bit, MTP K=7, single stream | quality reference |
| [`cyankiwi-seq8`](cyankiwi-seq8/) | + `max_num_seqs=8` | throughput; the **~3.4×** measured config |
| [`cyankiwi-seq8-mtp4`](cyankiwi-seq8-mtp4/) | + MTP K=4 + tuned CUDA graphs | latency (+8% local over seq8) |
| [`cyankiwi-seq8-mtp4-fp8kv`](cyankiwi-seq8-mtp4-fp8kv/) | + FP8 KV cache | memory headroom on 24 GB |

## Naming convention

`<weight-base>[-<delta>...]` — start from the weight provenance (`cyankiwi` =
the AWQ-4bit checkpoint) and append one suffix per optimization stacked on top,
so the name tells you what's in the variant. Date the *files* (e.g.
`latency_2026-05-25.json`), never the directory.

## Adding a variant

```bash
mkdir -p experiments/<name>
cp experiments/cyankiwi/config.env experiments/<name>/config.env   # then edit
# write experiments/<name>/README.md and measurements.json

make serve         VARIANT=<name>    # one terminal
make eval-latency  VARIANT=<name>    # another
make eval-quality  VARIANT=<name>
```

If a variant has no `config.env` or `weights/<name>/`, the Makefile falls back
to the `cyankiwi` reference.
