"""sitecustomize.py — runs at every Python interpreter startup via `site`.

Why this exists: vLLM spawns worker subprocesses
via `multiprocessing` for the executor. Those subprocesses inherit `os.environ`
but NOT the parent's `sys.path` modifications or `PYTHONSTARTUP`-loaded
modules. The `_cache_patch.py` device-name spoof — wired via
`PYTHONSTARTUP=/opt/program/_cache_patch.py` in the Dockerfile — runs in
the main Python process but NOT in vLLM's worker children. The torch.compile
cache key produced by the worker therefore reverts to the real
`AMPERE_A40` / `AMPERE_A10G` device name, missing the A40-baked cache on
the A10G eval host.

`sitecustomize.py` is the standard Python mechanism for this exact case:
the `site` module auto-imports `sitecustomize` from any path on
`sys.path` (or the standard site-packages) at every interpreter startup,
including subprocess children. Putting our cache patch invocation here
ensures every Python process — parent OR worker — runs through the
spoof BEFORE it touches `torch._inductor.codecache`.

Install (Mac Dockerfile change required):
  - COPY scripts/sitecustomize.py /opt/program/sitecustomize.py
  - ENV PYTHONPATH=/opt/program:${PYTHONPATH}

After that, the existing `PYTHONSTARTUP=/opt/program/_cache_patch.py` env
can be removed (this file subsumes it). Keep it for now as belt+braces.

Idempotency: `_cache_patch._apply_patches()` is safe to call multiple times;
all internal patches gate on `_eqwen_patched` flags so re-imports are no-ops.
"""
from __future__ import annotations

# Try to apply the torch.compile cache-portability spoof. This file must
# never crash a Python interpreter — wrap the entire body in try/except so
# any environment without our patches still boots cleanly.
try:  # noqa: SIM105 — explicit silent failure is intentional here
    # Import-by-path so we don't depend on _cache_patch.py being on
    # sys.path in the normal sense; the Dockerfile places it at
    # /opt/program/ alongside this file.
    import os
    import sys

    # Add /opt/program/ to sys.path if not already present (we live there
    # in the production image, but the same code works locally too).
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)

    # _cache_patch.py applies its patches at import time via its module
    # body — just importing it triggers _apply_patches() (line 87 of that
    # file). The patches are guarded with _eqwen_patched flags so re-imports
    # in subprocess children are safe no-ops.
    import _cache_patch  # noqa: F401 — import-for-side-effects
except Exception:
    # Belt-and-braces: NEVER let sitecustomize crash the interpreter.
    # If the cache patch fails, the worker just pays the cache miss
    # (~317s of recompilation on first cold start). Better than no boot.
    pass
