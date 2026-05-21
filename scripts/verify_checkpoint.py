#!/usr/bin/env python3
"""Sanity-check a downloaded checkpoint before docker build.

Verifies the basics — corrupted downloads, missing MTP heads, wrong
architecture — that would otherwise blow up at vLLM model-load time
inside the container (where the feedback loop is 90+ seconds).

Checks:
  - config.json exists and parses
  - architecture matches Qwen3.5 family
  - safetensors index references existing shard files
  - MTP head (mtp.fc) present — required for speculative decoding
  - chat_template.jinja or tokenizer_config.chat_template available
  - on-disk size ≤ Qwen3.5-4B BF16 base (~9.3 GB) per organizer rule

Exits 0 if all green, 1 if any error, 2 if warnings only.

Usage:
  python3 scripts/verify_checkpoint.py weights/cyankiwi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASE_MODEL_BYTES_LIMIT = int(9.3 * 1024**3)  # ~9.3 GB BF16 Qwen3.5-4B baseline
EXPECTED_ARCHS = {"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM",
                  "Qwen3_5MoeForConditionalGeneration"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("--require-mtp", action="store_true",
                   help="Fail (not warn) if MTP head is missing")
    args = p.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    if not args.src.is_dir():
        print(f"  ✗ {args.src} is not a directory", file=sys.stderr)
        return 1

    # 1. config.json
    cfg_path = args.src / "config.json"
    if not cfg_path.is_file():
        errors.append("config.json missing")
    else:
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as e:
            errors.append(f"config.json unparseable: {e}")
            cfg = {}

        archs = cfg.get("architectures", [])
        if not any(a in EXPECTED_ARCHS for a in archs):
            errors.append(f"architecture {archs} not in {sorted(EXPECTED_ARCHS)}")
        else:
            info.append(f"architecture: {archs[0]}")

    # 2. safetensors shards
    index_path = args.src / "model.safetensors.index.json"
    shards: list[Path] = []
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
        for name in shard_names:
            sp = args.src / name
            if not sp.is_file():
                errors.append(f"index references missing shard {name}")
            else:
                shards.append(sp)
    else:
        shards = sorted(args.src.glob("model*.safetensors"))
        if not shards:
            errors.append("no safetensors shards found and no index")

    info.append(f"shards: {len(shards)}")

    # 3. MTP head — scan first shard header (cheap, no full load)
    has_mtp = False
    if shards:
        import struct
        with open(shards[0], "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for key in header:
            if key.startswith("mtp."):
                has_mtp = True
                break
        # If multiple shards, also check the index (cheaper than reading each header)
        if not has_mtp and index_path.is_file():
            index = json.loads(index_path.read_text())
            if any(k.startswith("mtp.") for k in index.get("weight_map", {})):
                has_mtp = True

    if has_mtp:
        info.append("MTP head: present (speculative decoding ready)")
    else:
        msg = "MTP head (mtp.*) not found — speculative decoding unavailable on this checkpoint"
        if args.require_mtp:
            errors.append(msg)
        else:
            warnings.append(msg)

    # 4. Chat template
    has_template = (args.src / "chat_template.jinja").is_file()
    if not has_template:
        tok = args.src / "tokenizer_config.json"
        if tok.is_file():
            tcfg = json.loads(tok.read_text())
            has_template = bool(tcfg.get("chat_template"))
    if has_template:
        info.append("chat template: present")
    else:
        warnings.append("no chat_template.jinja and no tokenizer_config.chat_template "
                       "— /v1/chat/completions will fall back to vLLM default")

    # 5. Size budget
    total_bytes = sum(p.stat().st_size for p in shards)
    info.append(f"size: {total_bytes / 1024**3:.2f} GB "
               f"(budget: {BASE_MODEL_BYTES_LIMIT / 1024**3:.2f} GB)")
    if total_bytes > BASE_MODEL_BYTES_LIMIT:
        errors.append(f"checkpoint exceeds on-disk budget by "
                      f"{(total_bytes - BASE_MODEL_BYTES_LIMIT)/1024**2:.0f} MB")

    # Report
    for line in info:
        print(f"  ✓ {line}")
    for line in warnings:
        print(f"  ! {line}")
    for line in errors:
        print(f"  ✗ {line}")

    if errors:
        return 1
    if warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
