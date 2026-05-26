#!/usr/bin/env bash
# Build image.tar.gz — container artifact for the cyankiwi variant by default.
#
# Two modes (auto-selected):
#
#   MODE A: Docker available (Mac, dedicated GPU host)
#       Builds the Docker image from ./Dockerfile, then `docker save | gzip`
#       to image.tar.gz. The target host will `docker load` it.
#
#   MODE B: No Docker daemon (e.g. a GPU pod without a running docker daemon).
#       Streams files DIRECTLY into a single gzipped tarball via one tar
#       invocation piped to gzip — NO intermediate staging dir.
#       Tarball layout:
#           root/opt/ml/model/        — cyankiwi weights
#           root/opt/program/         — serve.py + _*_patch.py + run.sh
#           root/opt/venv/            — vLLM 0.19.0 .venv
#           manifest.json             — mode B marker + env dump
#       To convert Mode B → Docker image on a Docker host:
#           tar xzf image.tar.gz && cd root && \
#             docker build -t efficient-qwen:cyankiwi -f /path/to/Dockerfile .
#
# Outputs:  ./image.tar.gz at repo root (≤ 20 GB enforced)
# Env:
#   IMAGE         docker tag for mode A (default efficient-qwen:cyankiwi)
#   OUT           output tarball path (default image.tar.gz at repo root)
#   SKIP_BAKE     mode A: pass --build-arg SKIP_BAKE=1 (default 1)
#   FORCE_FS      force mode B even if docker exists (default 0)
#   DRY_RUN       if 1, prints what would happen but writes nothing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


IMAGE="${IMAGE:-efficient-qwen:cyankiwi}"
OUT="${OUT:-$REPO_ROOT/image.tar.gz}"
SKIP_BAKE="${SKIP_BAKE:-1}"
FORCE_FS="${FORCE_FS:-0}"
DRY_RUN="${DRY_RUN:-0}"

WEIGHTS_DIR="${WEIGHTS_DIR:-weights/cyankiwi}"
VENV_DIR="${VENV_DIR:-.venv}"

log() { printf "[build_image] %s\n" "$*" >&2; }

# ─── Pre-flight ─────────────────────────────────────────────────────────────
[[ -d "$WEIGHTS_DIR" ]]               || { log "FATAL: missing $WEIGHTS_DIR"; exit 1; }
[[ -f "$WEIGHTS_DIR/config.json" ]]   || { log "FATAL: $WEIGHTS_DIR/config.json missing"; exit 1; }
[[ -d "$VENV_DIR" ]]                  || { log "FATAL: missing $VENV_DIR"; exit 1; }
[[ -x "$VENV_DIR/bin/python3" ]]      || { log "FATAL: $VENV_DIR/bin/python3 missing"; exit 1; }
[[ -f scripts/serve.py ]]             || { log "FATAL: scripts/serve.py missing"; exit 1; }
[[ -f experiments/cyankiwi/config.env ]] || { log "FATAL: experiments/cyankiwi/config.env missing"; exit 1; }

# Decide mode
if command -v docker >/dev/null 2>&1 && [[ "$FORCE_FS" != "1" ]] \
     && docker info >/dev/null 2>&1; then
  MODE="A"
else
  MODE="B"
fi
log "MODE: $MODE (FORCE_FS=$FORCE_FS, docker=$(command -v docker || echo MISSING))"

# ─── Mode A: docker save | gzip ─────────────────────────────────────────────
if [[ "$MODE" == "A" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would: docker build --build-arg SKIP_BAKE=$SKIP_BAKE \\"
    log "[dry-run]              --build-arg WEIGHTS_DIR=$WEIGHTS_DIR -t $IMAGE ."
    log "[dry-run] would: docker save $IMAGE | gzip > $OUT"
    exit 0
  fi
  # CHAT_TEMPLATE_SRC env (optional) lets a variant bake a different chat
  # template into /opt/program/chat_template.jinja without forking the
  # Dockerfile. Pass-through to docker build only when set.
  EXTRA_BUILD_ARGS=()
  if [[ -n "${CHAT_TEMPLATE_SRC:-}" ]]; then
    EXTRA_BUILD_ARGS+=(--build-arg "CHAT_TEMPLATE_SRC=$CHAT_TEMPLATE_SRC")
  fi
  log "[1/2] docker build  (SKIP_BAKE=$SKIP_BAKE, WEIGHTS_DIR=$WEIGHTS_DIR, CHAT_TEMPLATE_SRC=${CHAT_TEMPLATE_SRC:-<default>})"
  docker build --platform linux/amd64 \
      --build-arg SKIP_BAKE="$SKIP_BAKE" \
      --build-arg WEIGHTS_DIR="$WEIGHTS_DIR" \
      "${EXTRA_BUILD_ARGS[@]}" \
      -t "$IMAGE" .
  log "[2/2] docker save | gzip → $OUT"
  docker save "$IMAGE" | gzip > "$OUT"
else
  # ─── Mode B: filesystem tarball (no Docker daemon) ────────────────────────
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] would: tar (weights+venv+program+manifest) | gzip → $OUT"
    log "[dry-run] would: transform paths to root/opt/{ml/model,venv,program}/"
    log "[dry-run] (no intermediate staging dir; one tar pass → gzip stream)"
    exit 0
  fi

  # Small files staged in a temp dir (program/ + manifest.json) so they can
  # join the same tar pass. The big stuff (weights/.venv) streams in via
  # --transform — no copy.
  WORK="$(mktemp -d -p "$REPO_ROOT" .build_work.XXXXXX)"
  trap 'rm -rf "$WORK"' EXIT

  log "[1/3] Stage small files (manifest, run.sh, serve + cache shims) in $WORK"
  mkdir -p "$WORK/program"
  cp scripts/serve.py          "$WORK/program/serve.py"
  cp scripts/_cache_patch.py   "$WORK/program/_cache_patch.py"
  cp scripts/sitecustomize.py  "$WORK/program/sitecustomize.py"
  cp scripts/bake_cache.py     "$WORK/program/bake_cache.py"

  cat > "$WORK/program/run.sh" <<'RUNSH'
#!/usr/bin/env bash
# Entrypoint for cyankiwi filesystem image. Mirrors Dockerfile ENV directives.
set -euo pipefail

# Offline mode (eval host has no internet)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/opt/ml/vllm_cache

# Compile-cache portability shim (AMPERE_SM86 device-name spoof so A10G eval host
# hits the same torch._inductor cache key as the A40 bake host).
# PYTHONSTARTUP loads in main process; PYTHONPATH addition makes
# /opt/program/sitecustomize.py auto-import in vLLM's worker subprocess
# children too (workers don't inherit PYTHONSTARTUP).
export PYTHONSTARTUP=/opt/program/_cache_patch.py
export PYTHONPATH=/opt/program:${PYTHONPATH:-}

# Cyankiwi production env (mirrors experiments/cyankiwi/config.env).
# GPU_MEMORY_UTILIZATION overridden 0.50 → 0.92 for target A10G (24 GB × 0.92).
export VLLM_LANGUAGE_MODEL_ONLY=1
export VLLM_MAX_NUM_SEQS=1
export VLLM_MAX_MODEL_LEN=13312
export VLLM_MAX_NUM_BATCHED_TOKENS=1024
export VLLM_BLOCK_SIZE=16
export VLLM_DTYPE=float16
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_ENABLE_PREFIX_CACHING=1
export VLLM_GPU_MEMORY_UTILIZATION=0.92
export VLLM_ENFORCE_EAGER=0
export VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":7}'
export VLLM_CUDAGRAPH_CAPTURE_SIZES=8
export VLLM_PORT=8080

# Use baked venv (vLLM 0.19.0 + all deps)
export PATH=/opt/venv/bin:$PATH

exec /opt/venv/bin/python3 /opt/program/serve.py
RUNSH
  chmod +x "$WORK/program/run.sh"

  cat > "$WORK/manifest.json" <<MANIFEST
{
  "format": "filesystem-tarball-v1",
  "schema": "efficient-qwen-2026",
  "image_tag_intended": "$IMAGE",
  "entrypoint": ["/opt/program/run.sh"],
  "port": 8080,
  "model_dir": "/opt/ml/model",
  "venv_dir": "/opt/venv",
  "program_dir": "/opt/program",
  "weights_variant": "cyankiwi",
  "variant": "${VARIANT:-cyankiwi}",
  "built_on": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "built_by_host": "$(hostname)"
}
MANIFEST

  # gzip flavor: parallel if pigz available
  if command -v pigz >/dev/null 2>&1; then
    GZ="pigz -p $(nproc)"
    log "Using pigz ($(nproc) threads)"
  else
    GZ="gzip"
    log "Using single-thread gzip (no pigz available — expect ~10-20 min for ~21 GB raw)"
  fi

  # Single tar pass with multi-source paths + --transform rewriting each
  # source's prefix to its in-image location, piped straight to gzip. No
  # intermediate uncompressed tar on disk.
  #
  # --transform expressions are evaluated in order; the FIRST matching one
  # wins. We anchor with ^ to guarantee root-level prefix matching.
  log "[2/3] Streaming tar → gzip → $OUT  (one pass, no intermediate file)"
  # Compute WORK relative to REPO_ROOT so the --transform anchor is stable
  # (tar stores names relative to cwd).
  WORK_REL="${WORK#$REPO_ROOT/}"
  tar -c --hard-dereference \
      --transform "s,^$WEIGHTS_DIR,root/opt/ml/model,Sx" \
      --transform "s,^$VENV_DIR,root/opt/venv,Sx" \
      --transform "s,^$WORK_REL/program,root/opt/program,Sx" \
      --transform "s,^$WORK_REL/manifest\.json\$,manifest.json,Sx" \
      --exclude='.cache' --exclude='__pycache__' --exclude='*.pyc' \
      "$WEIGHTS_DIR" "$VENV_DIR" "$WORK_REL/program" "$WORK_REL/manifest.json" \
  | $GZ > "$OUT"

  log "[3/3] Cleanup"
  rm -rf "$WORK"
  trap - EXIT
fi

# ─── Verify size + report ───────────────────────────────────────────────────
SIZE=$(wc -c < "$OUT" | tr -d ' ')
SIZE_GB=$(awk -v s="$SIZE" 'BEGIN{printf "%.2f", s/1024/1024/1024}')
log "image.tar.gz: ${SIZE_GB} GB  (cap 20 GB)"
if [[ "$SIZE" -gt 21474836480 ]]; then
  log "FATAL: exceeds 20 GB cap"
  exit 1
fi

echo "$OUT"
