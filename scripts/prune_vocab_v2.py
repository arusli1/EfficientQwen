#!/usr/bin/env python3
"""Vocab pruning — tied-embeddings path for cyankiwi.

Produces a vocabulary-pruned checkpoint plus a token-ID remap sidecar that
scripts/serve.py consumes via VLLM_VOCAB_REMAP_SIDECAR. Notes:
  * Tied embeddings: no `lm_head.weight` key exists. Only slice
    `model.language_model.embed_tokens.weight`. vLLM ties LM head at load.
  * No dependency on the (nonexistent) `cyankiwi_untied_bf16/`.
  * Emits `model.safetensors.index.json` (single-shard, like the source).
  * MTP tensors (mtp.*) pass through unchanged — they don't index vocab.
  * Sidecar `orig_to_new_token_ids.json` is list-form with -1 for dropped,
    per spec. Mirrored as `pruned_token_map.json` for legacy callers.
  * Default --keep-strategy first_n_plus_special: keep IDs
    [0 .. target - n_specials) plus all 33 tokenizer specials. The on-disk
    BPE vocab is NOT renumbered; runtime remap at the API layer handles any
    rare token > new_vocab_size.

Run:
  .venv/bin/python scripts/prune_vocab_v2.py \\
      --target-vocab 64000 --src weights/cyankiwi --dst weights/cyankiwi-v64k
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

EMBED_KEY = "model.language_model.embed_tokens.weight"
ORIG_VOCAB = 248320  # cyankiwi text_config.vocab_size

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "preprocessor_config.json",
    "video_preprocessor_config.json", "generation_config.json",
)


def load_special_ids(src: Path) -> list[int]:
    """All added_tokens_decoder IDs, sorted (no filtering)."""
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    return sorted(int(tid) for tid in cfg.get("added_tokens_decoder", {}))


def build_keep_ids_first_n_plus_special(
    target_vocab: int, specials: list[int]
) -> list[int]:
    """Keep IDs [0 .. target - n_specials) plus all specials. Returns sorted."""
    n_spec = len(specials)
    if n_spec >= target_vocab:
        raise ValueError(f"target_vocab={target_vocab} <= n_specials={n_spec}")
    head = list(range(target_vocab - n_spec))
    keep = sorted(set(head) | set(specials))
    if len(keep) != target_vocab:
        raise ValueError(
            f"keep size {len(keep)} != target {target_vocab} "
            f"(special/head collision? specials_min={min(specials)})"
        )
    return keep


def build_keep_ids_frequency(
    target_vocab: int, specials: list[int], repo_root: Path
) -> list[int]:
    """Frequency-based keep set. Needs id_frequencies in vocab_coverage.json."""
    cov_path = repo_root / "results" / "vocab_coverage.json"
    if not cov_path.exists():
        raise FileNotFoundError(f"frequency strategy needs {cov_path}")
    cov = json.loads(cov_path.read_text())
    freqs = cov.get("id_frequencies") or cov.get("token_id_counts")
    if not freqs:
        raise NotImplementedError(
            "vocab_coverage.json has no 'id_frequencies'; "
            "use --keep-strategy first_n_plus_special"
        )
    spec_set = set(specials)
    ranked = sorted(
        ((int(k), int(v)) for k, v in freqs.items() if int(k) not in spec_set),
        key=lambda kv: (-kv[1], kv[0]),
    )
    n_data = target_vocab - len(specials)
    data_ids = [tid for tid, _ in ranked[:n_data]]
    if len(data_ids) < n_data:
        used = spec_set | set(data_ids)
        for tid in range(ORIG_VOCAB):
            if tid not in used:
                data_ids.append(tid)
                if len(data_ids) >= n_data:
                    break
    keep = sorted(set(data_ids) | spec_set)
    if len(keep) != target_vocab:
        raise ValueError(f"frequency keep size {len(keep)} != {target_vocab}")
    return keep


def build_sidecar(keep_ids: list[int]) -> dict:
    """{orig_to_new: [...], new_to_orig: [...]}, -1 sentinel for dropped."""
    orig_to_new = [-1] * ORIG_VOCAB
    for new_id, orig in enumerate(keep_ids):
        orig_to_new[orig] = new_id
    return {
        "orig_to_new": orig_to_new,
        "new_to_orig": list(keep_ids),
        "schema_version": 2,
        "target_vocab": len(keep_ids),
        "orig_vocab": ORIG_VOCAB,
    }


def write_pruned_safetensors(
    src_weights_path: Path, dst_path: Path, keep_ids: list[int]
) -> tuple[dict[str, str], list[str]]:
    """Slice embed_tokens rows; pass everything else (incl. mtp.*) through."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    keep_idx = torch.tensor(keep_ids, dtype=torch.long)
    new_tensors: dict[str, "torch.Tensor"] = {}
    summary: dict[str, str] = {}
    with safe_open(str(src_weights_path), framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if key == EMBED_KEY:
                if t.dim() != 2 or t.shape[0] != ORIG_VOCAB:
                    raise ValueError(
                        f"unexpected embed shape {tuple(t.shape)} for {key}"
                    )
                new_t = t.index_select(0, keep_idx).contiguous()
                summary[key] = f"{tuple(t.shape)} -> {tuple(new_t.shape)}"
                new_tensors[key] = new_t
            else:
                new_tensors[key] = t
    save_file(new_tensors, str(dst_path))
    return summary, list(new_tensors.keys())


def verify_safetensors(dst_path: Path, expected_rows: int) -> None:
    from safetensors import safe_open
    with safe_open(str(dst_path), framework="pt") as f:
        if EMBED_KEY not in f.keys():
            raise RuntimeError(f"{EMBED_KEY} missing from written file")
        t = f.get_tensor(EMBED_KEY)
        if t.shape[0] != expected_rows:
            raise RuntimeError(f"embed rows {t.shape[0]} != {expected_rows}")


def rewrite_config(src: Path, dst: Path, target_vocab: int) -> None:
    """Update vocab_size at top-level AND text_config. Don't touch token IDs:
    first_n_plus_special keeps ALL specials so eos/bos/pad/image/video_token_id
    are unchanged. tie_word_embeddings passes through (True)."""
    cfg = json.loads((src / "config.json").read_text())
    if "vocab_size" in cfg:
        cfg["vocab_size"] = target_vocab
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"]["vocab_size"] = target_vocab
    else:
        print("[warn] config.json missing text_config dict", file=sys.stderr)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def write_index(dst: Path, weights_name: str, keys: list[str]) -> None:
    idx = {"metadata": {}, "weight_map": {k: weights_name for k in keys}}
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))


def copy_tokenizer_files(src: Path, dst: Path) -> list[str]:
    copied = []
    for name in TOKENIZER_FILES:
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, dst / name)
            copied.append(name)
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prune vocab on cyankiwi (tied embeds).")
    ap.add_argument("--target-vocab", type=int, default=64000)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument(
        "--keep-strategy",
        choices=("first_n_plus_special", "frequency"),
        default="first_n_plus_special",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src, dst = args.src.resolve(), args.dst.resolve()
    if not src.is_dir():
        print(f"[err] --src not a dir: {src}", file=sys.stderr)
        return 2
    if dst.exists() and not args.dry_run:
        print(f"[err] --dst exists; refusing overwrite: {dst}", file=sys.stderr)
        return 2
    src_weights = src / "model-00001-of-00001.safetensors"
    if not src_weights.exists():
        print(f"[err] safetensors not found: {src_weights}", file=sys.stderr)
        return 2

    specials = load_special_ids(src)
    print(f"[info] {len(specials)} specials, IDs {specials[0]}..{specials[-1]}")

    if args.keep_strategy == "first_n_plus_special":
        keep_ids = build_keep_ids_first_n_plus_special(args.target_vocab, specials)
    else:
        repo_root = src.parent if (src.parent / "results").is_dir() else src
        keep_ids = build_keep_ids_frequency(args.target_vocab, specials, repo_root)

    sidecar = build_sidecar(keep_ids)
    n_kept_spec = sum(1 for s in specials if sidecar["orig_to_new"][s] >= 0)
    print(f"[plan] keep={len(keep_ids)} dropped={ORIG_VOCAB - len(keep_ids)} "
          f"strategy={args.keep_strategy} specials_kept={n_kept_spec}/{len(specials)}")

    if args.dry_run:
        print("[dry-run] would write:")
        print(f"  {dst}/model-00001-of-00001.safetensors  (embed -> ({args.target_vocab}, 2560))")
        print(f"  {dst}/model.safetensors.index.json")
        print(f"  {dst}/config.json  (vocab_size={args.target_vocab} top + text_config)")
        print(f"  {dst}/orig_to_new_token_ids.json  (sidecar)")
        print(f"  {dst}/pruned_token_map.json  (alias)")
        print(f"  tokenizer files copied: {list(TOKENIZER_FILES)}")
        return 0

    dst.mkdir(parents=True, exist_ok=False)
    dst_weights = dst / "model-00001-of-00001.safetensors"
    print(f"[apply] writing -> {dst_weights}")
    summary, all_keys = write_pruned_safetensors(src_weights, dst_weights, keep_ids)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("[apply] verifying reload...")
    verify_safetensors(dst_weights, expected_rows=args.target_vocab)

    print("[apply] writing index / config / sidecar / tokenizer files")
    write_index(dst, dst_weights.name, all_keys)
    rewrite_config(src, dst, args.target_vocab)
    (dst / "orig_to_new_token_ids.json").write_text(json.dumps(sidecar))
    (dst / "pruned_token_map.json").write_text(json.dumps(sidecar))
    print(f"[apply] tokenizer files copied: {copy_tokenizer_files(src, dst)}")
    print(f"[done] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
