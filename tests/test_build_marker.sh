#!/usr/bin/env bash
# tests/test_build_marker.sh — direct shell test of scripts/lib/build_marker.sh.
#
# Usage:
#   bash tests/test_build_marker.sh
#
# Also driven by tests/test_build_marker.py via pytest so this hooks into the
# project's `make test` target without requiring pytest-bash plumbing.
#
# Test approach: spin up a throwaway git repo in a temp dir, point
# BUILD_MARKER_DIR + BUILD_MARKER_REPO_ROOT at it, write fake markers + a fake
# image.tar.gz, then assert each of the failure modes from the spec.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$REPO_ROOT/scripts/lib/build_marker.sh"

PASS=0
FAIL=0
FAILED_CASES=()

ok()   { PASS=$((PASS+1)); printf "  PASS  %s\n" "$*"; }
bad()  { FAIL=$((FAIL+1)); FAILED_CASES+=("$*"); printf "  FAIL  %s\n" "$*"; }

# Spin a fresh sandbox per case: temp dir + tiny git repo + fake image.
new_sandbox() {
  local d
  d="$(mktemp -d)"
  ( cd "$d" \
    && git init -q \
    && git config user.email "t@t" \
    && git config user.name "t" \
    && echo x > x \
    && git add x \
    && git commit -q -m init )
  mkdir -p "$d/artifacts"
  # 1 MiB fake "image"
  dd if=/dev/zero of="$d/image.tar.gz" bs=1024 count=1024 status=none
  echo "$d"
}

# Re-source the lib with our sandbox roots — variables are module-global so we
# can't just call functions; we have to override and re-source.
run_verify() {
  local sandbox="$1"; shift
  bash -c "
    set -uo pipefail
    export BUILD_MARKER_REPO_ROOT='$sandbox'
    export BUILD_MARKER_DIR='$sandbox/artifacts'
    . '$LIB'
    # Override REPO_ROOT *after* sourcing too (lib's bootstrap sets it from
    # its own location; the env vars above take precedence in the functions).
    BUILD_MARKER_REPO_ROOT='$sandbox'
    BUILD_MARKER_DIR='$sandbox/artifacts'
    verify_build_marker $*
  " 2>&1
}

run_write() {
  local sandbox="$1"; local mode="$2"; local variant="$3"
  bash -c "
    set -uo pipefail
    export BUILD_MARKER_REPO_ROOT='$sandbox'
    export BUILD_MARKER_DIR='$sandbox/artifacts'
    . '$LIB'
    BUILD_MARKER_REPO_ROOT='$sandbox'
    BUILD_MARKER_DIR='$sandbox/artifacts'
    write_build_marker '$sandbox/image.tar.gz' '$mode' '$variant'
  " 2>&1
}

# Build a marker file by hand (override fields the write helper would compute).
# Args: sandbox commit_sha date_yyyymmdd build_date_utc size sha
write_fake_marker() {
  local sandbox="$1" commit="$2" date_short="$3" build_date="$4" size="$5" sha="$6"
  local short="${commit:0:7}"
  local out="$sandbox/artifacts/build_${short}_${date_short}.md"
  cat > "$out" <<EOF
# Build marker — $short @ $date_short

commit: $commit
commit_date_utc: 2026-05-25T00:00:00+00:00
build_date_utc: $build_date
variant: test-variant
image_path: image.tar.gz
image_size_bytes: $size
image_sha256: $sha
build_mode: filesystem-tarball
vllm_version: 0.19.0
EOF
  printf "%s" "$out"
}

# ─── 1. No marker → verify fails with right message ─────────────────────────
case1() {
  local d; d="$(new_sandbox)"
  local out; out="$(run_verify "$d")"; local rc=$?
  if [[ $rc -ne 0 ]] && grep -q "no build marker found" <<<"$out"; then
    ok "case 1: no marker → fails with 'no build marker found'"
  else
    bad "case 1: rc=$rc out=$out"
  fi
  rm -rf "$d"
}

# ─── 2. Marker dated yesterday → stale ─────────────────────────────────────
case2() {
  local d; d="$(new_sandbox)"
  local sha head size
  head="$(git -C "$d" rev-parse HEAD)"
  size="$(wc -c < "$d/image.tar.gz" | tr -d ' ')"
  sha="$(shasum -a 256 "$d/image.tar.gz" 2>/dev/null | awk '{print $1}')"
  [[ -z "$sha" ]] && sha="$(sha256sum "$d/image.tar.gz" | awk '{print $1}')"
  # macOS BSD date vs GNU date — use python for reliable yesterday.
  local yesterday yesterday_short
  yesterday="$(python3 -c 'import datetime;print((datetime.datetime.utcnow()-datetime.timedelta(days=1)).strftime("%Y-%m-%dT12:00:00Z"))')"
  yesterday_short="$(python3 -c 'import datetime;print((datetime.datetime.utcnow()-datetime.timedelta(days=1)).strftime("%Y%m%d"))')"
  write_fake_marker "$d" "$head" "$yesterday_short" "$yesterday" "$size" "$sha" >/dev/null
  local out; out="$(run_verify "$d")"; local rc=$?
  if [[ $rc -ne 0 ]] && grep -q "stale build" <<<"$out"; then
    ok "case 2: yesterday marker → stale-build message"
  else
    bad "case 2: rc=$rc out=$out"
  fi
  rm -rf "$d"
}

# ─── 3. Marker SHA != HEAD → commit mismatch ───────────────────────────────
case3() {
  local d; d="$(new_sandbox)"
  local size sha today today_short
  size="$(wc -c < "$d/image.tar.gz" | tr -d ' ')"
  sha="$(shasum -a 256 "$d/image.tar.gz" 2>/dev/null | awk '{print $1}')"
  [[ -z "$sha" ]] && sha="$(sha256sum "$d/image.tar.gz" | awk '{print $1}')"
  today="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  today_short="$(date -u +%Y%m%d)"
  # Use a fake SHA that's not HEAD.
  write_fake_marker "$d" "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef" \
    "$today_short" "$today" "$size" "$sha" >/dev/null
  local out; out="$(run_verify "$d")"; local rc=$?
  if [[ $rc -ne 0 ]] && grep -q "build commit .* != HEAD" <<<"$out"; then
    ok "case 3: wrong-SHA marker → commit-mismatch message"
  else
    bad "case 3: rc=$rc out=$out"
  fi
  rm -rf "$d"
}

# ─── 4. Marker correct → verify passes ─────────────────────────────────────
case4() {
  local d; d="$(new_sandbox)"
  local out; out="$(run_write "$d" "filesystem-tarball" "test-variant")"
  if [[ $? -ne 0 ]]; then
    bad "case 4 (write): out=$out"; rm -rf "$d"; return
  fi
  out="$(run_verify "$d")"; local rc=$?
  if [[ $rc -eq 0 ]] && grep -q "build marker OK" <<<"$out"; then
    ok "case 4: correct marker → verify passes"
  else
    bad "case 4 (verify): rc=$rc out=$out"
  fi
  rm -rf "$d"
}

# ─── 5. Size mismatch (image truncated after marker written) ───────────────
case5() {
  local d; d="$(new_sandbox)"
  run_write "$d" "filesystem-tarball" "test-variant" >/dev/null
  # Truncate the image to a different size.
  dd if=/dev/zero of="$d/image.tar.gz" bs=1024 count=512 status=none
  local out; out="$(run_verify "$d")"; local rc=$?
  if [[ $rc -ne 0 ]] && grep -q "image size mismatch" <<<"$out"; then
    ok "case 5: size-changed image → size-mismatch message"
  else
    bad "case 5: rc=$rc out=$out"
  fi
  rm -rf "$d"
}

# ─── 6. --fast skips sha256 (but still catches size mismatch) ──────────────
# We can't easily "mock a slow sha256" portably, but the substantive promise of
# --fast is "skip the sha256 verification step". We exercise this by:
#   (a) writing a correct marker, then mutating the image's CONTENT (not size)
#       so sha256 differs but size matches.
#   (b) confirming default verify fails on sha (sha-mismatch message), but
#       --fast passes.
case6() {
  local d; d="$(new_sandbox)"
  run_write "$d" "filesystem-tarball" "test-variant" >/dev/null
  # Same size, different content (overwrite byte 0).
  local sz; sz="$(wc -c < "$d/image.tar.gz" | tr -d ' ')"
  dd if=/dev/urandom of="$d/image.tar.gz" bs="$sz" count=1 status=none
  # Default verify should fail on sha mismatch.
  local out_slow; out_slow="$(run_verify "$d")"; local rc_slow=$?
  if [[ $rc_slow -ne 0 ]] && grep -q "sha256 mismatch" <<<"$out_slow"; then
    : # expected
  else
    bad "case 6 (default): expected sha mismatch, rc=$rc_slow out=$out_slow"
    rm -rf "$d"; return
  fi
  # --fast verify should pass (skips sha256 check).
  local out_fast; out_fast="$(run_verify "$d" --fast)"; local rc_fast=$?
  if [[ $rc_fast -eq 0 ]] && grep -q "build marker OK" <<<"$out_fast"; then
    ok "case 6: --fast skips sha256 (passes despite sha mismatch)"
  else
    bad "case 6 (--fast): rc=$rc_fast out=$out_fast"
  fi
  rm -rf "$d"
}

case1
case2
case3
case4
case5
case6

printf "\n%s\n" "------------------------------------------------------------"
printf "build_marker.sh tests: %d passed, %d failed\n" "$PASS" "$FAIL"
if (( FAIL > 0 )); then
  printf "failed cases:\n"
  for c in "${FAILED_CASES[@]}"; do printf "  - %s\n" "$c"; done
  exit 1
fi
exit 0
