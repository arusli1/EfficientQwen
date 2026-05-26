FROM adaptfm/adaptfm-base:latest
# Runtime deps come from the base image (vLLM 0.19.0, transformers, FastAPI,
# uvicorn, lm-eval 0.4.11, CUDA 12.4). The target runtime is sealed (no
# internet), so don't `pip install` here — vendor any extras into the image.

# Bake weights at /opt/ml/model/. Populate ./weights/<variant>/ first:
#   make download   (or:  python3 scripts/download_weights.py --variant cyankiwi)
ARG WEIGHTS_DIR=weights/cyankiwi
COPY ${WEIGHTS_DIR}/ /opt/ml/model/

# Per-variant chat template baked at /opt/program/chat_template.jinja.
# Defaults to the model's own template; a variant can bake a different one:
#   docker build --build-arg CHAT_TEMPLATE_SRC=experiments/<v>/chat_template.jinja
# scripts/serve.py picks this up via VLLM_CHAT_TEMPLATE (set below).
ARG CHAT_TEMPLATE_SRC=weights/cyankiwi/chat_template.jinja
COPY ${CHAT_TEMPLATE_SRC} /opt/program/chat_template.jinja

# Custom serve.py overlays the base image's serve_default.py with our flags.
COPY scripts/serve.py /opt/program/serve.py

# Cold-start fix: spoofs the GPU device name to AMPERE_SM86 so the
# torch._inductor cache key matches between A40 (build host) and A10G
# (target host). Pre-bakes the torch.compile cache at build time so first
# boot hits the warm-cache fast path (~156s vs 697s naive).
COPY scripts/_cache_patch.py /opt/program/_cache_patch.py
COPY scripts/bake_cache.py /opt/program/bake_cache.py
# sitecustomize.py runs at every Python interpreter startup via `site`,
# INCLUDING vLLM's multiprocessing worker subprocess children. The
# PYTHONSTARTUP path below loads _cache_patch in the main process only;
# the workers re-init torch._inductor and would miss the cache key
# otherwise (vLLM worker subprocesses don't inherit PYTHONSTARTUP).
# Belt-and-braces with PYTHONSTARTUP below.
COPY scripts/sitecustomize.py /opt/program/sitecustomize.py

# Target runtime has no internet
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_CACHE_ROOT=/opt/ml/vllm_cache \
    PYTHONSTARTUP=/opt/program/_cache_patch.py \
    PYTHONPATH=/opt/program:${PYTHONPATH:-}

# Baseline serve config. Target runtime can't pass env at startup, so these
# MUST live in the image. VLLM_QUANTIZATION stays unset — cyankiwi is
# compressed-tensors (Marlin-routed by auto-detect); setting awq_marlin
# would crash the load.
#
# VLLM_GPU_MEMORY_UTILIZATION=0.92 is sized for the A10G (24 GB × 0.92 = 22 GB).
# Lower it only when serving on a larger card under emulation/testing.
ENV VLLM_LANGUAGE_MODEL_ONLY=1 \
    VLLM_MAX_NUM_SEQS=8 \
    VLLM_MAX_MODEL_LEN=13312 \
    VLLM_MAX_NUM_BATCHED_TOKENS=1024 \
    VLLM_BLOCK_SIZE=16 \
    VLLM_DTYPE=float16 \
    VLLM_ENABLE_CHUNKED_PREFILL=1 \
    VLLM_ENABLE_PREFIX_CACHING=1 \
    VLLM_GPU_MEMORY_UTILIZATION=0.92 \
    VLLM_ENFORCE_EAGER=0 \
    VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":4}' \
    VLLM_CUDAGRAPH_CAPTURE_SIZES=5,10,15,20,25,30,35,40 \
    VLLM_REPETITION_PENALTY=1.0 \
    VLLM_CHAT_TEMPLATE=/opt/program/chat_template.jinja
# Bakes the cyankiwi-seq8-mtp4 config: MTP K=4 + max_num_seqs=8 + matched
# CUDA-graph capture sizes (K=4 candidates × {1..8} sequences). The capture
# set determines the torch._inductor cache key, so the pre-baked
# cache_import.tar.gz must be produced with the same env to hit the warm path.

# Env wins from vLLM source audit. TREE_ATTN was tried but is incompatible
# with FULL_AND_PIECEWISE cudagraph at capture_sizes=[8] (forces PIECEWISE
# only → loses cudagraph capture → net negative). 4 remaining flags are safe:
#   VLLM_COMPILATION_CONFIG  qk_norm + RoPE fusion (expected +3-8% decode)
#   VLLM_ATTENTION_CONFIG    cudnn_prefill only, NO TREE_ATTN (+2-6% TTFT)
#   FLA_USE_CUDA_GRAPH       enables cudagraph in GDN kernel (+5-10%)
#   FLA_GDN_FIX_BT           bumps GDN chunk-block dim to 64 (+1-3%)
# Combined: +5-15% wall TPS. enable_qk_norm_rope_fusion may shift the
# torch._inductor cache key → rebake cache_import.tar.gz if so. If OOM/SIGKILL
# at init, bisect by setting flags to "" one at a time.
ENV VLLM_COMPILATION_CONFIG='{"pass_config":{"enable_qk_norm_rope_fusion":true}}' \
    VLLM_ATTENTION_CONFIG='{"use_cudnn_prefill":true}' \
    FLA_USE_CUDA_GRAPH=1 \
    FLA_GDN_FIX_BT=1

# Populate the torch.compile cache. Three modes (in priority order):
#   1. IMPORT (default if cache_import.tar.gz is non-empty): COPY a cache
#      tarball pre-built on a GPU host. Works on Mac without --gpus.
#   2. NATIVE BAKE (SKIP_BAKE=0, no import tarball): launch vLLM at build time
#      to populate the cache. Requires `docker build --gpus all` (GPU host).
#   3. SKIP (SKIP_BAKE=1): do nothing. Image will have ~697s cold-start.
#
# An empty placeholder cache_import.tar.gz is committed at repo root so this
# COPY always succeeds. If you want the IMPORT path, replace it with a real
# tarball produced by `python3 scripts/bake_cache.py` on a GPU host with
# the device-name shim active (PYTHONSTARTUP=scripts/_cache_patch.py).
COPY cache_import.tar.gz /opt/ml/cache_import.tar.gz

ARG SKIP_BAKE=0
RUN mkdir -p /opt/ml/vllm_cache && \
    if [ -s /opt/ml/cache_import.tar.gz ] && [ $(stat -c%s /opt/ml/cache_import.tar.gz) -gt 1024 ]; then \
      echo "Importing pre-built torch.compile cache from cache_import.tar.gz" && \
      tar xzf /opt/ml/cache_import.tar.gz -C /opt/ml/vllm_cache/ && \
      du -sh /opt/ml/vllm_cache/ ; \
    elif [ "$SKIP_BAKE" = "0" ]; then \
      echo "Running native bake (requires --gpus all)" && \
      python3 /opt/program/bake_cache.py && \
      du -sh /opt/ml/vllm_cache/ ; \
    else \
      echo "SKIP_BAKE=1 and no import tarball; cold-start will be ~697s" ; \
    fi && \
    rm -f /opt/ml/cache_import.tar.gz

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/ping || exit 1

ENTRYPOINT ["python3", "/opt/program/serve.py"]
