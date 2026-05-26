#!/usr/bin/env bash
# Inspect a built submission image's structure — works without GPU.
# Catches "I forgot to bake X" + "weights at wrong path" bugs cheaply.
#
# Usage:  scripts/verify_image.sh [image_tag]   (default: efficient-qwen:dev)

set -euo pipefail
IMG="${1:-efficient-qwen:dev}"

ok() { printf "  \033[32m✓\033[0m  %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m  %s\n" "$1"; FAILED=1; }
FAILED=0

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "image $IMG not found — run: make build"; exit 1; }

# Image size cap
SIZE_BYTES=$(docker image inspect "$IMG" --format='{{.Size}}')
SIZE_GB=$(awk -v s="$SIZE_BYTES" 'BEGIN{printf "%.2f", s/1024/1024/1024}')
if [[ "$SIZE_BYTES" -lt 21474836480 ]]; then
  ok "size ${SIZE_GB} GB (cap 20)"
else
  fail "size ${SIZE_GB} GB EXCEEDS 20 GB cap"
fi

# Entrypoint
ENTRY=$(docker image inspect "$IMG" --format='{{join .Config.Entrypoint " "}}')
if [[ "$ENTRY" == *"serve.py"* ]]; then
  ok "ENTRYPOINT: $ENTRY"
else
  fail "ENTRYPOINT unexpected: $ENTRY"
fi

# Port exposed
EXPOSED=$(docker image inspect "$IMG" --format='{{range $p,$_ := .Config.ExposedPorts}}{{$p}} {{end}}')
[[ "$EXPOSED" == *"8080"* ]] && ok "port 8080 exposed" || fail "port 8080 NOT exposed (saw: $EXPOSED)"

# Inside the image
docker run --rm --platform linux/amd64 --entrypoint sh "$IMG" -c '
errs=0
check() { eval "$1" >/dev/null 2>&1 && echo "  ✓ $2" || { echo "  ✗ $2"; errs=$((errs+1)); }; }

check "test -f /opt/program/serve.py"                      "/opt/program/serve.py present"
check "python3 -c \"import ast; ast.parse(open(\\\"/opt/program/serve.py\\\").read())\"" "serve.py parses"
check "test -f /opt/ml/model/config.json"                  "/opt/ml/model/config.json present"
check "test -f /opt/ml/model/chat_template.jinja"          "/opt/ml/model/chat_template.jinja present"
check "ls /opt/ml/model/model*.safetensors >/dev/null"     "safetensors shard(s) present"
check "python3 -c \"import json; assert json.load(open(\\\"/opt/ml/model/config.json\\\"))[\\\"architectures\\\"][0].startswith(\\\"Qwen3_5\\\")\"" "config.json architecture is Qwen3_5*"
check "[ \"\$TRANSFORMERS_OFFLINE\" = \"1\" ]"             "TRANSFORMERS_OFFLINE=1"
check "[ \"\$HF_HUB_OFFLINE\" = \"1\" ]"                   "HF_HUB_OFFLINE=1"
check "[ -z \"\$VLLM_QUANTIZATION\" ]"                     "VLLM_QUANTIZATION unset (cyankiwi is compressed-tensors; awq_marlin would crash load)"
check "[ \"\$VLLM_LANGUAGE_MODEL_ONLY\" = \"1\" ]"         "VLLM_LANGUAGE_MODEL_ONLY=1"
check "[ \"\$VLLM_MAX_MODEL_LEN\" = \"13312\" ]"           "VLLM_MAX_MODEL_LEN=13312 (8448 silently broke GPQA — never regress)"
check "[ -n \"\$VLLM_SPECULATIVE_CONFIG\" ]"               "VLLM_SPECULATIVE_CONFIG set (no MTP = image is 1×)"
check "[ \"\$VLLM_CUDAGRAPH_CAPTURE_SIZES\" = \"8\" ]"     "VLLM_CUDAGRAPH_CAPTURE_SIZES=8 (Path B contraction — must match the baked cache key)"
exit $errs
' || FAILED=1

[[ "$FAILED" == 1 ]] && { echo; echo "VERIFY FAILED"; exit 1; }
echo
echo "Image $IMG looks good for submission."
