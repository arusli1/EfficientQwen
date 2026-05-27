"""Smoke test — every Python file in scripts/ must import without error.

Catches syntax errors / missing imports before A10G time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module — Python 3.9's dataclass
    # decorator calls `sys.modules.get(cls.__module__).__dict__` to resolve
    # annotations; if the module isn't registered, None.__dict__ raises
    # AttributeError. Affects any script with `from __future__ import
    # annotations` + @dataclass + lowercase generics (list[int], dict[k,v]).
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


@pytest.mark.parametrize("path", sorted(Path("scripts").glob("*.py")))
def test_script_imports(path):
    _load(path)
