#!/usr/bin/env bash
# profile_matrix.sh — run scripts/profile_model.py across short/medium/long
# prompt categories for ONE variant. Writes per-category JSONs + a summary
# aggregating verdicts under results/profile/<variant>/.
#
# Usage:
#   scripts/profile_matrix.sh <variant> [container-url]
#   VARIANT=cyankiwi-seq8 scripts/profile_matrix.sh
#
# Categories map to scripts/profile_model.py --prompt-lens (matches
# PROMPT_CONFIGS in eval_common.py: short=64, medium=2048, long=8192).

set -euo pipefail

VARIANT="${1:-${VARIANT:-}}"
URL="${2:-${CONTAINER_URL:-http://localhost:8080}}"
[ -n "$VARIANT" ] || { echo "usage: $0 <variant> [container-url]"; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

OUT_DIR="$REPO/results/profile/$VARIANT"
mkdir -p "$OUT_DIR"

declare -A LEN=([short]=64 [medium]=2048 [long]=8192)

echo "=== profile_matrix | variant=$VARIANT | url=$URL | out=$OUT_DIR ==="
for cat in short medium long; do
  out="$OUT_DIR/op_${cat}.json"
  echo ">>> [$cat] prompt_len=${LEN[$cat]} -> $out"
  "$PY" "$REPO/scripts/profile_model.py" \
    --model-name "${VARIANT}-${cat}" \
    --container-url "$URL" \
    --prompt-lens "${LEN[$cat]}" \
    --out "$out"
done

# Aggregate verdicts into summary.json (stdlib only — pod is sealed).
"$PY" - <<EOF
import json, os, time
out_dir = "$OUT_DIR"
summary = {"variant": "$VARIANT", "container_url": "$URL",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "categories": {}}
for cat in ("short", "medium", "long"):
    p = os.path.join(out_dir, f"op_{cat}.json")
    if not os.path.isfile(p):
        summary["categories"][cat] = {"error": "missing"}
        continue
    with open(p) as f:
        d = json.load(f)
    summary["categories"][cat] = {
        "decode_share_overall": d.get("decode_share_overall"),
        "kv_cache_usage_peak": d.get("kv_cache_usage_peak"),
        "iteration_tokens_per_sec": d.get("iteration_tokens_per_sec"),
        "slope_prefill_ms_per_1k": d.get("slope_prefill_ms_per_1k"),
        "slope_decode_intercept_ms": d.get("slope_decode_intercept_ms"),
        "mtp_by_category": d.get("mtp_by_category"),
        "verdict": d.get("verdict"),
        "wall_total_s": d.get("wall_total_s"),
    }
levers = [c.get("verdict", {}).get("next_lever") for c in summary["categories"].values()
          if isinstance(c, dict) and c.get("verdict")]
summary["lever_consensus"] = (levers[0] if levers and all(L == levers[0] for L in levers)
                              else "MIXED" if levers else "UNKNOWN")
out_path = os.path.join(out_dir, "summary.json")
tmp = out_path + ".tmp"
with open(tmp, "w") as f:
    json.dump(summary, f, indent=2)
os.replace(tmp, out_path)
print(f"[summary] wrote {out_path}  lever_consensus={summary['lever_consensus']}")
EOF
