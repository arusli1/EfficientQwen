#!/usr/bin/env python3
"""Download Qwen3.5-4B weights.

Variants:
  cyankiwi   — AWQ 4-bit checkpoint used by this project (default)
  paro       — z-lab/Qwen3.5-4B-PARO (rotation INT4)
  quanttrio  — QuantTrio/Qwen3.5-4B-AWQ (community AWQ INT4)
  base       — original BF16 Qwen/Qwen3.5-4B, for a self-quantization path

Writes to ./weights/<variant>/. Pin a revision with --revision for
reproducibility.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

VARIANTS = {
    "cyankiwi": "cyankiwi/Qwen3.5-4B-AWQ-4bit",
    "paro": "z-lab/Qwen3.5-4B-PARO",
    "quanttrio": "QuantTrio/Qwen3.5-4B-AWQ",
    "base": "Qwen/Qwen3.5-4B",
}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--variant", choices=sorted(VARIANTS), default="cyankiwi")
    p.add_argument("--dest", type=Path, default=None,
                   help="Destination dir (default: ./weights/<variant>)")
    p.add_argument("--revision", default=None,
                   help="Pin to a specific HF revision/commit (recommended before submission)")
    args = p.parse_args()

    repo_id = VARIANTS[args.variant]
    dest = args.dest or Path("weights") / args.variant
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {dest}", file=sys.stderr)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        revision=args.revision,
        ignore_patterns=["*.gguf", "*.bin.index.json.lock", ".gitattributes"],
    )
    print(f"Done. Weights at: {dest.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
