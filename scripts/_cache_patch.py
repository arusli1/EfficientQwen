#!/usr/bin/env python3
"""Device-name spoof for torch.compile cache portability.

Makes A40 (cache bake host) and A10G (eval host) produce the same inductor
cache key by overriding the GPU device name string that
``torch._inductor.codecache.CacheBase.get_system()`` hashes into compiler_factors.
Without this, the in-image baked cache misses on target A10G and we pay the
full ~317s torch.compile cost on cold-start.

Activated via ``PYTHONSTARTUP=/opt/program/_cache_patch.py`` in the Dockerfile,
so it loads at every Python process start (including vLLM's subprocesses).
Idempotent: no-op if torch isn't importable, no CUDA, or already patched.

Two patch paths are applied (belt-and-braces):
  1. Override torch.cuda.get_device_properties to mutate the wrapper's .name.
     C++ binding; may silently no-op on some torch versions.
  2. Override torch._inductor.codecache.CacheBase.get_system to drop the
     device.name key from the returned dict. Survives even if (1) doesn't stick.

Selftest:
  python3 scripts/_cache_patch.py --selftest
  → device_name=AMPERE_SM86      (PASS — primary path works)
  → FALLBACK_OK: device.name dropped from CacheBase.get_system()  (PASS — secondary works)
  → FAIL                          (both paths broken — will MISS cache)
"""
from __future__ import annotations

SPOOF_NAME = "AMPERE_SM86"


def _apply_patches() -> None:
    """Apply both spoof paths. Safe to call multiple times."""
    try:
        import torch
    except ImportError:
        return  # noop for non-torch processes (don't crash unrelated python invocations)

    if not getattr(torch.cuda, "is_available", lambda: False)():
        return  # noop on CPU-only hosts (Mac dev, CI)

    _patch_device_properties(torch)
    _patch_inductor_cache_base()


def _patch_device_properties(torch) -> None:
    if getattr(torch.cuda.get_device_properties, "_eqwen_patched", False):
        return
    real = torch.cuda.get_device_properties

    import contextlib

    def patched(idx):
        p = real(idx)
        # Fall through to secondary path silently if the C++ binding rejects setattr.
        with contextlib.suppress(Exception):
            object.__setattr__(p, "name", SPOOF_NAME)
        return p

    patched._eqwen_patched = True  # type: ignore[attr-defined]
    torch.cuda.get_device_properties = patched


def _patch_inductor_cache_base() -> None:
    try:
        from torch._inductor.codecache import CacheBase
    except Exception:
        return  # older torch / minimal install — secondary path unavailable

    if getattr(getattr(CacheBase.get_system, "__func__", CacheBase.get_system),
               "_eqwen_patched", False):
        return

    real = CacheBase.get_system

    @staticmethod
    def patched():
        info = real()
        if isinstance(info, dict) and isinstance(info.get("device"), dict):
            dev = {k: v for k, v in info["device"].items() if k != "name"}
            info = {**info, "device": dev}
        return info

    patched.__func__._eqwen_patched = True  # type: ignore[attr-defined]
    CacheBase.get_system = patched


_apply_patches()


def _selftest() -> int:
    import sys
    try:
        import torch
    except ImportError:
        print("FAIL: torch not importable", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device (this script's effect can only be verified on a GPU host)",
              file=sys.stderr)
        return 0

    props = torch.cuda.get_device_properties(0)
    name = props.name
    print(f"device_name={name}")
    if name == SPOOF_NAME:
        return 0

    try:
        from torch._inductor.codecache import CacheBase
        info = CacheBase.get_system()
        if isinstance(info, dict) and isinstance(info.get("device"), dict):
            if "name" not in info["device"]:
                print("FALLBACK_OK: device.name dropped from CacheBase.get_system()")
                return 0
            print(f"FALLBACK_FAIL: device still has name={info['device']['name']!r}",
                  file=sys.stderr)
    except Exception as e:
        print(f"FALLBACK_FAIL: {e}", file=sys.stderr)

    print(f"FAIL: device_name still {name!r}; cache key will differ A40 vs A10G",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Patches applied at import. Use --selftest on a CUDA host to verify.")
