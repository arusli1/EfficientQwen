#!/usr/bin/env python3
"""Lightweight JSON schema check for EfficientQwen artifacts.

Two loose schemas (required-key presence only, no value validation):

  Schema A — experiments/<variant>/measurements.json
    required: schema_version, variant, weights_origin, status

  Schema B — results/<name>.json
    required: schema_version, measurement_type, date, verdict

Run:
  python scripts/check_schemas.py           # scan both default trees
  python scripts/check_schemas.py --path results/foo.json
  python scripts/check_schemas.py --path experiments/

Exit 0 if all pass, 1 if any errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA_A_REQUIRED = ("schema_version", "variant", "weights_origin", "status")
SCHEMA_B_REQUIRED = ("schema_version", "measurement_type", "date", "verdict")

# Schema A.gate: optional isolation_gate.baseline_* fields used as a pre-flight
# regression check (compare new variant to declared baseline). The reference
# variant (cyankiwi-seq8) is exempt because IT IS the baseline. Other variants
# either declare baselines or set skip_pass_a_diagnostic=true (audit trail).
ISOLATION_GATE_BASELINE_KEYS = (
    "baseline_variant",
    "baseline_mmlu_pro",
    "baseline_ifeval",
    "baseline_gpqa_d",
)
EXEMPT_VARIANTS = frozenset({"cyankiwi", "cyankiwi-seq8"})

# Schema C: chat-completions latency bench output.
# Includes endpoint + thinking_mode discriminators per category.
SCHEMA_C_PER_CAT_REQUIRED = ("median_ms", "n", "speedup_vs_baseline",
                             "endpoint", "thinking_mode")

OK = "✓"
BAD = "✗"


def classify(path: Path) -> str | None:
    """Return 'A', 'B', 'C', or None based on path location/content."""
    parts = path.resolve().parts
    name = path.name
    if name == "measurements.json" and "experiments" in parts:
        return "A"
    # latency_chat_*.json under experiments/ is Schema C (chat-bench output)
    if "experiments" in parts and name.startswith("latency_chat_"):
        return "C"
    if "results" in parts and path.suffix == ".json":
        return "B"
    return None


def _check_isolation_gate(data: dict, variant: str) -> list[str]:
    """Variants outside EXEMPT_VARIANTS must declare isolation_gate.baseline_*
    OR isolation_gate.skip_pass_a_diagnostic=true (audit trail required).
    Returns list of error strings (empty = OK)."""
    errors: list[str] = []
    if variant in EXEMPT_VARIANTS:
        return errors
    # Status-based exemption: scaffolded/blocked variants don't need baselines
    # yet (they haven't been benched). Once a variant has a measurement, it
    # must declare its baseline reference for the external regression check.
    status = (data.get("status") or "").lower()
    if status in ("scaffolded", "blocked", "smoke_partial"):
        return errors
    gate = data.get("isolation_gate")
    if not isinstance(gate, dict):
        errors.append(
            "missing isolation_gate dict (variant beyond scaffold needs "
            "baseline declared OR skip_pass_a_diagnostic=true)"
        )
        return errors
    if gate.get("skip_pass_a_diagnostic") is True:
        return errors  # explicit opt-out, audit-trailed
    missing = [k for k in ISOLATION_GATE_BASELINE_KEYS if k not in gate]
    if missing:
        errors.append(
            f"isolation_gate missing baseline_* keys: {', '.join(missing)} "
            "(needed by external regression check; set "
            "skip_pass_a_diagnostic=true to opt out)"
        )
    return errors


def _check_chat_bench(data: dict) -> list[str]:
    """Schema C: chat-completions latency bench output. Per-category dicts
    under {short, medium, long} keys (overall is OK to omit per-cat fields)."""
    errors: list[str] = []
    for cat in ("short", "medium", "long"):
        if cat not in data:
            continue  # bench may run a subset of cats
        cat_data = data[cat]
        if not isinstance(cat_data, dict):
            errors.append(f"{cat}: must be dict, got {type(cat_data).__name__}")
            continue
        missing = [k for k in SCHEMA_C_PER_CAT_REQUIRED if k not in cat_data]
        if missing:
            errors.append(f"{cat}: missing required keys: {', '.join(missing)}")
        # endpoint must be 'chat' for a chat-completions bench output
        if cat_data.get("endpoint") not in (None, "chat"):
            errors.append(
                f"{cat}.endpoint={cat_data['endpoint']!r}; expected 'chat' "
                "(chat-completions bench output)"
            )
    return errors


def check_file(path: Path) -> tuple[bool, list[str], list[str]]:
    """Return (ok, errors, warnings).

    `errors` block exit 0 (schema-correctness violations). `warnings` are
    informational — surfaced but don't fail the run (e.g., isolation_gate
    baselines missing — the real enforcement is the external regression check).
    """
    errors: list[str] = []
    warnings: list[str] = []
    kind = classify(path)
    if kind is None:
        return True, [f"skipped (unrecognized location): {path}"], []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return False, [f"invalid JSON: {exc}"], []
    except OSError as exc:
        return False, [f"read error: {exc}"], []
    if not isinstance(data, dict):
        return False, [f"top-level must be object, got {type(data).__name__}"], []
    if kind == "A":
        missing = [k for k in SCHEMA_A_REQUIRED if k not in data]
        if missing:
            errors.append(f"missing required keys (A): {', '.join(missing)}")
        # isolation_gate is a WARNING — the external regression check is the
        # actual enforcement point. Missing baselines = "shipped without
        # local quality verification". Doesn't break the schema.
        warnings.extend(_check_isolation_gate(data, data.get("variant", "")))
    elif kind == "B":
        missing = [k for k in SCHEMA_B_REQUIRED if k not in data]
        if missing:
            errors.append(f"missing required keys (B): {', '.join(missing)}")
    elif kind == "C":
        errors.extend(_check_chat_bench(data))
    return (not errors), errors, warnings


def collect(path: Path) -> list[Path]:
    """Resolve --path arg into a concrete file list."""
    if path.is_file():
        return [path]
    if path.is_dir():
        files: list[Path] = []
        # measurements.json under experiments/*/
        files.extend(sorted(path.glob("experiments/*/measurements.json")))
        # any .json under results/
        files.extend(sorted(path.glob("results/*.json")))
        # latency_chat_*.json under experiments/*/ (Schema C)
        files.extend(sorted(path.glob("experiments/*/latency_chat_*.json")))
        # if path itself is experiments/ or results/
        if path.name == "experiments":
            files.extend(sorted(path.glob("*/measurements.json")))
            files.extend(sorted(path.glob("*/latency_chat_*.json")))
        elif path.name == "results":
            files.extend(sorted(path.glob("*.json")))
        return sorted(set(files))
    return []


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--path", type=Path, default=REPO_ROOT,
                    help="file or directory to scan (default: repo root)")
    args = ap.parse_args()

    files = collect(args.path)
    if not files:
        print(f"no schema-bound JSON files found under {args.path}")
        return 0

    fail = 0
    warn_count = 0
    for f in files:
        ok, errs, warns = check_file(f)
        rel = f.relative_to(REPO_ROOT) if REPO_ROOT in f.parents or f == REPO_ROOT else f
        if ok:
            if warns:
                print(f"  {OK} {rel}  ({len(warns)} warning{'s' if len(warns) != 1 else ''})")
                for w in warns:
                    print(f"      WARN: {w}")
                warn_count += len(warns)
            else:
                print(f"  {OK} {rel}")
        else:
            fail += 1
            print(f"  {BAD} {rel}")
            for e in errs:
                print(f"      {e}")
    total = len(files)
    summary = f"\n{total - fail}/{total} ok"
    if fail:
        summary += f", {fail} failed"
    if warn_count:
        summary += f", {warn_count} warning{'s' if warn_count != 1 else ''}"
    print(summary)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
