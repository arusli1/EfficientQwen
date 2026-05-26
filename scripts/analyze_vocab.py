#!/usr/bin/env python3
"""Vocab coverage analysis — unblocks A2 (vocab prune) target-N decision.

Tokenizes the FROZEN eval task datasets (MMLU-Pro + IFEval + GPQA-Diamond)
plus optional calibration prompts, computes the union of token IDs that
ever appear (prompt OR expected completion), adds the tokenizer's special
tokens, and reports candidate prune targets with safety margins.

Output: results/vocab_coverage.json + console summary.

Failure modes are non-fatal: a dataset that errors is skipped and noted in
the JSON so the decision-maker sees the gap.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def _tokenize(tok, text: str) -> set[int]:
    if not isinstance(text, str) or not text.strip():
        return set()
    try:
        return set(tok.encode(text))
    except Exception:
        try:
            return set(tok(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return set()


def _fields_of(rec: dict) -> Iterable[str]:
    """Pull every plausible prompt/completion field out of a HF row.

    MMLU-Pro: question + options + answer (letter usually, but tokenize anyway)
    IFEval:   prompt
    GPQA:     Question + Correct Answer + 3 distractors
    Generic:  text / prompt / content / response / target / answer
    """
    for k in ("question", "Question", "prompt", "Prompt", "text",
              "content", "instruction", "input", "context"):
        v = rec.get(k)
        if isinstance(v, str):
            yield v
    for k in ("answer", "Answer", "Correct Answer", "answer_text",
              "target", "response", "output", "completion"):
        v = rec.get(k)
        if isinstance(v, str):
            yield v
    # MMLU-Pro options is list[str]
    opts = rec.get("options")
    if isinstance(opts, list):
        for o in opts:
            if isinstance(o, str):
                yield o
    # GPQA distractors
    for k in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"):
        v = rec.get(k)
        if isinstance(v, str):
            yield v
    # GPQA Explanation (long text — feeds many tokens, may be in prompt)
    expl = rec.get("Explanation")
    if isinstance(expl, str):
        yield expl


def _load_dataset(path: str, name: str | None, split: str):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import load_dataset
    return load_dataset(path, name) if name else load_dataset(path)


def _coverage_for_dataset(label: str, dataset, tok,
                          counts: collections.Counter,
                          notes: list[str]) -> dict:
    n_rows = 0
    n_skipped = 0
    local: set[int] = set()
    for split_name, split in dataset.items():
        for rec in split:
            n_rows += 1
            row_ids: set[int] = set()
            for field_text in _fields_of(rec):
                row_ids |= _tokenize(tok, field_text)
            if not row_ids:
                n_skipped += 1
                continue
            local |= row_ids
            for t in row_ids:
                counts[t] += 1
    notes.append(f"  {label}: {n_rows} rows, {n_skipped} empty, "
                 f"{len(local)} unique tokens")
    return {"label": label, "rows": n_rows, "rows_empty": n_skipped,
            "unique_tokens": len(local)}


def _coverage_for_calib(path: Path, tok, counts: collections.Counter,
                        notes: list[str]) -> dict:
    if not path.is_file():
        notes.append(f"  calib({path}): MISSING — skipped")
        return {"label": "calib_1024", "missing": True}
    local: set[int] = set()
    n_rows = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            for k in ("text", "prompt", "content"):
                v = rec.get(k)
                if isinstance(v, str):
                    ids = _tokenize(tok, v)
                    local |= ids
                    for t in ids:
                        counts[t] += 1
    notes.append(f"  calib_1024: {n_rows} rows, {len(local)} unique tokens")
    return {"label": "calib_1024", "rows": n_rows,
            "unique_tokens": len(local)}


def _safety_margin(union: int, n: int) -> str:
    if n <= union:
        return f"INSUFFICIENT — drops {union - n} required tokens"
    return f"covers union × {n / max(union, 1):.2f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", default="weights/cyankiwi",
                   help="HF model dir for tokenizer.")
    p.add_argument("--calib", default="data/calibration_1024.jsonl")
    p.add_argument("--out", default="results/vocab_coverage.json")
    args = p.parse_args()

    print(f"[vocab] loading tokenizer from {args.weights}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.weights)
    vocab_size = tok.vocab_size
    # Many Qwen tokenizers add tokens above vocab_size (the 'added_tokens').
    # The full encodable space is len(tok) which includes those.
    full_vocab = len(tok)
    print(f"[vocab] tokenizer vocab_size={vocab_size} len(tok)={full_vocab}",
          flush=True)

    counts: collections.Counter = collections.Counter()
    notes: list[str] = []
    per_dataset = []

    datasets_spec = [
        ("mmlu_pro", "TIGER-Lab/MMLU-Pro", None),
        ("ifeval", "wis-k/instruction-following-eval", None),
        ("gpqa_diamond", "Idavidrein/gpqa", "gpqa_diamond"),
    ]
    for label, path, name in datasets_spec:
        print(f"[vocab] loading {label} ({path})", flush=True)
        try:
            ds = _load_dataset(path, name, split=None)
            per_dataset.append(
                _coverage_for_dataset(label, ds, tok, counts, notes))
        except Exception as e:
            msg = f"  {label}: FAILED to load: {e}"
            print("[vocab]" + msg, flush=True)
            notes.append(msg)
            per_dataset.append({"label": label, "error": str(e)})

    per_dataset.append(
        _coverage_for_calib(Path(args.calib), tok, counts, notes))

    # Union = every token id that appeared anywhere.
    union = set(counts.keys())
    # Add special-token IDs (BOS/EOS/PAD/UNK/extra) — never prune these.
    special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id",
                 "unk_token_id", "sep_token_id", "cls_token_id",
                 "mask_token_id"):
        v = getattr(tok, attr, None)
        if isinstance(v, int):
            special_ids.add(v)
    # additional_special_tokens
    for st in (tok.additional_special_tokens or []):
        sid = tok.convert_tokens_to_ids(st)
        if isinstance(sid, int):
            special_ids.add(sid)
    # full added-token map
    for tok_str, tok_id in (tok.get_added_vocab() or {}).items():
        special_ids.add(tok_id)
    print(f"[vocab] special token ids: {sorted(special_ids)[:20]}"
          f"{'…' if len(special_ids) > 20 else ''} (n={len(special_ids)})",
          flush=True)
    union |= special_ids

    union_n = len(union)
    print(f"\n[vocab] UNION size: {union_n} (of full vocab {full_vocab})",
          flush=True)

    # Frequency histogram in coarse buckets.
    freq_buckets = {
        "1": 0, "2-10": 0, "11-100": 0,
        "101-1000": 0, "1001-10000": 0, "10000+": 0,
    }
    for tok_id, c in counts.items():
        if c == 1:
            freq_buckets["1"] += 1
        elif c <= 10:
            freq_buckets["2-10"] += 1
        elif c <= 100:
            freq_buckets["11-100"] += 1
        elif c <= 1000:
            freq_buckets["101-1000"] += 1
        elif c <= 10000:
            freq_buckets["1001-10000"] += 1
        else:
            freq_buckets["10000+"] += 1

    candidate_ns = sorted({max(union_n, 32000), 64000, 96000, 128000})
    candidates = [
        {"N": n, "safety": _safety_margin(union_n, n),
         "savings_vs_full": f"{(1 - n/full_vocab) * 100:.1f}%"}
        for n in candidate_ns
    ]

    out = {
        "tokenizer": {"path": args.weights, "vocab_size": vocab_size,
                      "len_full": full_vocab},
        "datasets": per_dataset,
        "special_token_count": len(special_ids),
        "union_size": union_n,
        "frequency_histogram": freq_buckets,
        "candidate_targets": candidates,
        "recommendation": _recommendation(union_n, full_vocab),
        "notes": notes,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[vocab] wrote {args.out}", flush=True)

    print("\n" + "=" * 70)
    print("VOCAB COVERAGE SUMMARY")
    print("=" * 70)
    print(f"  union of tokens used: {union_n} (full vocab {full_vocab})")
    print(f"  reduction headroom: {(1 - union_n/full_vocab) * 100:.1f}% of vocab is unused on this eval mix")
    print("  candidate prune targets:")
    for c in candidates:
        print(f"    N={c['N']:>7}  · {c['safety']:<30}  · saves {c['savings_vs_full']} of vocab")
    print(f"  recommendation: {out['recommendation']}")
    return 0


def _recommendation(union_n: int, full_vocab: int) -> str:
    """Conservative recommendation: smallest N ≥ union × 2 that's a round-ish
    number. Doubling provides margin for unseen prompts at cloud eval time.
    """
    target = max(union_n * 2, 32000)
    for candidate in (32000, 48000, 64000, 96000, 128000, 192000):
        if candidate >= target:
            return (f"N={candidate} (≥ 2× observed union of {union_n}; "
                    f"saves {(1 - candidate/full_vocab) * 100:.1f}% vs full)")
    return f"no candidate ≥ 2× union of {union_n}; full vocab needed"


if __name__ == "__main__":
    sys.exit(main())
