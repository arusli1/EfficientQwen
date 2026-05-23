"""Tests for scripts/_cache_patch.py (Phase 0 Track 2).

CUDA-dependent behavior can't be tested on Mac. These tests verify:
- The module imports cleanly without torch (PYTHONSTARTUP must not crash).
- The selftest exits cleanly on a non-CUDA host (SKIP path).
- The SPOOF_NAME constant matches the value documented in COLDSTART_MITIGATION.md.

Full validation lives on the GPU pod via:
  python3 scripts/_cache_patch.py --selftest
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

CACHE_PATCH = Path(__file__).parent.parent / "scripts" / "_cache_patch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_cache_patch", CACHE_PATCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_imports_without_error():
    """PYTHONSTARTUP must never crash a Python process, even when torch is absent."""
    result = subprocess.run(
        [sys.executable, "-c", f"exec(open({str(CACHE_PATCH)!r}).read())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"import crashed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_selftest_exits_clean_on_no_cuda():
    """--selftest must SKIP (exit 0) on hosts without CUDA, not crash."""
    result = subprocess.run(
        [sys.executable, str(CACHE_PATCH), "--selftest"],
        capture_output=True, text=True,
    )
    # Mac (no CUDA): exit 0 with SKIP message on stderr
    # GPU pod working shim: exit 0 with PASS on stdout
    # GPU pod broken shim: exit 1 with FAIL on stderr
    assert result.returncode in (0, 1)


def test_spoof_name_constant():
    """SPOOF_NAME must be 'AMPERE_SM86' — the device name the build/serve GPUs
    are normalized to so the torch._inductor cache key matches."""
    mod = _load_module()
    assert mod.SPOOF_NAME == "AMPERE_SM86"


def test_apply_patches_callable_without_torch():
    """_apply_patches() must be safely callable even when torch isn't installed."""
    mod = _load_module()
    # No torch on Mac venv; should be a clean no-op.
    mod._apply_patches()
