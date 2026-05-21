.PHONY: help install install-eval download verify datasets build build-nobake \
        build-import verify-image \
        test lint serve eval-quality eval-quality-full eval-latency eval-chat \
        eval-smoke profile-matrix check-schemas clean distclean

help:  ## Show this help.
	@awk 'BEGIN{FS=":.*?## "}/^[a-zA-Z_-]+:.*?## /{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}' $(MAKEFILE_LIST)

# ─── Setup ────────────────────────────────────────────────────────────────────
install:  ## Create .venv and install host-side deps
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements-dev.txt
	@echo "Activate with: source .venv/bin/activate"

install-eval:  ## Add heavy eval deps (datasets + lm-eval) — GPU box only
	.venv/bin/pip install -q -r requirements-eval.txt

download:  ## Download cyankiwi weights to ./weights/cyankiwi/ (~3.8 GB)
	.venv/bin/python scripts/download_weights.py --variant cyankiwi

verify:  ## Sanity-check downloaded checkpoint
	.venv/bin/python scripts/verify_checkpoint.py weights/cyankiwi

datasets:  ## First-time only — download eval datasets (eval scripts default to offline)
	HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/python -c "\
from datasets import load_dataset; \
load_dataset('TIGER-Lab/MMLU-Pro'); \
load_dataset('wis-k/instruction-following-eval'); \
load_dataset('Idavidrein/gpqa', 'gpqa_diamond')"

# ─── Develop ──────────────────────────────────────────────────────────────────
test:  ## Run the pytest suite (~6s, no GPU needed)
	.venv/bin/pytest tests/ -q

lint:  ## Static analysis (ruff)
	.venv/bin/ruff check scripts/ tests/

# ─── Build ────────────────────────────────────────────────────────────────────
build:  ## docker build the container image (GPU host needed for native bake)
	@test -d weights/$(VARIANT) || test -d weights/cyankiwi || { echo "Missing weights — run: make download"; exit 1; }
	docker build --build-arg WEIGHTS_DIR=$(_VWEIGHTS) -t efficient-qwen:$(VARIANT) .

build-nobake:  ## docker build WITHOUT prebake — image works but cold-start is 697s (debugging / Mac-only)
	docker build --platform linux/amd64 --build-arg SKIP_BAKE=1 \
	  -t efficient-qwen:cyankiwi . 2>&1 | tee /tmp/docker_build.log

build-import:  ## docker build using a pre-built compile cache (cache_import.tar.gz at repo root)
	@test -s cache_import.tar.gz && [ $$(stat -f%z cache_import.tar.gz 2>/dev/null || stat -c%s cache_import.tar.gz) -gt 1024 ] \
	  || { echo "cache_import.tar.gz missing or empty — produce by running scripts/bake_cache.py on a GPU host then tarring /opt/ml/vllm_cache/"; exit 1; }
	docker build --platform linux/amd64 --build-arg SKIP_BAKE=1 \
	  -t efficient-qwen:cyankiwi . 2>&1 | tee /tmp/docker_build.log
	@echo "Verify cache imported: docker run --rm --entrypoint sh efficient-qwen:cyankiwi -c 'du -sh /opt/ml/vllm_cache/'"

verify-image:  ## Inspect a built image's structure (no GPU needed; works on Mac under emulation)
	@scripts/verify_image.sh efficient-qwen:$(VARIANT)

# ─── Variants ─────────────────────────────────────────────────────────────────
# Each variant lives in experiments/<name>/ (config.env + README + measurement
# JSONs). Weights at weights/<name>/. Falls back to cyankiwi if a piece is missing.
# Naming: <weight-base>[-<delta>] — `cyankiwi` is the AWQ-4bit reference;
# suffixes describe each optimization stacked on top (e.g. cyankiwi-seq8).
#   make serve         VARIANT=cyankiwi      (default)
#   make eval-quality  VARIANT=cyankiwi-seq8
VARIANT ?= cyankiwi
_VWEIGHTS = $(shell test -d weights/$(VARIANT) && echo weights/$(VARIANT) || echo weights/cyankiwi)
_VCFG = $(shell test -f experiments/$(VARIANT)/config.env && echo experiments/$(VARIANT)/config.env || echo experiments/cyankiwi/config.env)
_VDIR = experiments/$(VARIANT)
_TODAY := $(shell date -u +%Y-%m-%d)

serve:  ## Start vLLM directly (no docker). Override: make serve VARIANT=name
	@test -d $(_VWEIGHTS) || { echo "Missing $(_VWEIGHTS) — run: make download (or generate weights/$(VARIANT)/)"; exit 1; }
	@test -x .venv/bin/python || { echo "Missing .venv — run: make install"; exit 1; }
	@echo "Serving variant=$(VARIANT)  weights=$(_VWEIGHTS)  config=$(_VCFG)"
	set -a; . $(_VCFG); set +a; \
	VLLM_MODEL=$(PWD)/$(_VWEIGHTS) \
	VLLM_NO_USAGE_STATS=1 \
	PYTHONSTARTUP=$(PWD)/scripts/_cache_patch.py \
	.venv/bin/python scripts/serve.py

# ─── Local eval (run against a running container OR `make serve`; wait for /ping first) ────
# QUALITY: uses eval/run_quality_local.py (lm-eval-harness driver against the
#   running vLLM via OpenAI chat-completions). Same per-task config across runs.
# LATENCY: uses scripts/bench_latency.py (diverse natural prompts, no
#   prefix-cache repeat). Outputs land in experiments/<VARIANT>/<task>_<date>.json
#   (also tee'd to /tmp).

eval-quality:  ## Quality eval at 10% sample (~20 min, ~5-10pp noise vs full)
	@which lm_eval >/dev/null 2>&1 || { echo "Missing lm_eval — run: make install-eval"; exit 1; }
	@mkdir -p $(_VDIR)
	QUALITY_LIMIT=0.1 NUM_CONCURRENT=8 \
	OUTPUT_PATH=$(_VDIR)/quality_quick_$(_TODAY).json \
	.venv/bin/python eval/run_quality_local.py 2>&1 | tee /tmp/quality_quick.log

eval-quality-full:  ## Quality eval at 100% sample (~60 min, definitive)
	@which lm_eval >/dev/null 2>&1 || { echo "Missing lm_eval — run: make install-eval"; exit 1; }
	@mkdir -p $(_VDIR)
	QUALITY_LIMIT=1.0 NUM_CONCURRENT=8 \
	OUTPUT_PATH=$(_VDIR)/quality_full_$(_TODAY).json \
	.venv/bin/python eval/run_quality_local.py 2>&1 | tee /tmp/quality_full.log

eval-latency:  ## Latency probe on diverse natural prompts
	@mkdir -p $(_VDIR)
	OUTPUT_PATH=$(_VDIR)/latency_$(_TODAY).json \
	.venv/bin/python scripts/bench_latency.py 2>&1 | tee /tmp/latency.log

eval-chat:  ## Chat-completions latency probe (mirrors the /v1/chat path)
	@mkdir -p $(_VDIR)
	OUTPUT_PATH=$(_VDIR)/latency_chat_$(_TODAY).json \
	.venv/bin/python scripts/bench_chat_latency.py 2>&1 | tee /tmp/latency_chat.log

# ─── Quick-data layer (sub-eval-quality smoke + op-level profiling matrix) ────
eval-smoke:  ## 5-prompts/task quality smoke (~3-5 min, ungated — sanity only)
	@mkdir -p $(_VDIR)
	.venv/bin/python scripts/eval_smoke.py \
	  --model-name $(VARIANT) \
	  --out $(_VDIR)/smoke_$(_TODAY).json 2>&1 | tee /tmp/smoke.log

profile-matrix:  ## Run scripts/profile_model.py across short+medium+long → results/profile/$(VARIANT)/
	scripts/profile_matrix.sh $(VARIANT)

check-schemas:  ## Validate result-JSON schemas across results/ + experiments/
	.venv/bin/python scripts/check_schemas.py

# ─── Housekeeping ─────────────────────────────────────────────────────────────
clean:  ## Remove image tarball, caches, /tmp logs (PRESERVES experiments/)
	rm -f image.tar.gz
	rm -rf .pytest_cache .ruff_cache
	rm -f /tmp/quality_quick.log /tmp/quality_full.log /tmp/latency.log /tmp/smoke.log /tmp/docker_build.log
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean  ## Also remove weights/ and .venv/ (forces full re-setup; KEEPS experiments/)
	rm -rf weights/ .venv/
