"""Pytest wrapper around tests/test_build_marker.sh — picks up under `make test`.

The actual assertions live in the bash script (closer to the bash lib under
test). This thin wrapper just shells out and surfaces pass/fail counts.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "test_build_marker.sh"


@pytest.mark.timeout(60)
def test_build_marker_lib():
    """Run the shell-driven build_marker test cases.

    See tests/test_build_marker.sh for the six cases. Each spins up a throwaway
    git repo in /tmp and exercises one failure / success mode of the lib's
    verify_build_marker function.
    """
    assert SCRIPT.exists(), f"missing {SCRIPT}"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        # Surface the script's own pass/fail summary to make failures readable.
        pytest.fail(
            f"build_marker.sh tests failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
        )
