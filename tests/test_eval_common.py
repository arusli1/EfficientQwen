"""CPU-only tests for scripts/eval_common.py + the two thin wrappers.

We don't spin up a real container — networked behavior is covered by the
end-to-end runs. These tests catch the cheap regressions: import works,
constants match the FROZEN eval/run_quality_local.py, CLI flags parse,
limit defaults are what the leaderboard expects, helpers are pure.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def _load(modname: str, relpath: str):
    spec = importlib.util.spec_from_file_location(modname, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def common():
    # eval_full/eval_fast import `from eval_common import …`, so eval_common
    # must be importable as a top-level module. Inject scripts/ on sys.path.
    sys.path.insert(0, str(REPO / "scripts"))
    return _load("eval_common", "scripts/eval_common.py")


def test_quality_tasks_match_frozen_eval(common):
    # Source of truth: eval/run_quality_local.py — must not drift.
    expected = [
        ("mmlu_pro",                  "mmlu_pro",     5, "exact_match,custom-extract",   0.621, False),
        ("ifeval",                    "ifeval",       0, "inst_level_strict_acc,none",   0.814, False),
        ("gpqa_diamond_cot_zeroshot", "gpqa_diamond", 0, "exact_match,flexible-extract", 0.630, True),
    ]
    assert common.QUALITY_TASKS == expected


def test_baseline_latency_matches_competition(common):
    # CLAUDE.md "Per-category baselines" row.
    assert common.BASELINE_LATENCY_MS == {
        "short": 2582, "medium": 5441, "long": 6576,
    }


def test_utc_stamp_format(common):
    s = common.utc_stamp()
    # ISO-8601 with Z suffix; constant length 20.
    assert len(s) == 20
    assert s.endswith("Z")
    assert s[10] == "T"


def test_shape_pads_short_and_truncates_long(common):
    short_in = "abc"
    out = common._shape(short_in, tokens=10)
    assert len(out) == 40  # tokens * 4 chars/token
    long_in = "x" * 1000
    out = common._shape(long_in, tokens=10)
    assert out == "x" * 40


def test_load_calib_returns_none_when_missing(common, monkeypatch):
    monkeypatch.setattr(common, "CALIB_PATH", "/nonexistent/path.jsonl")
    assert common._load_calib_prompts() is None


def test_load_calib_reads_jsonl(common, monkeypatch, tmp_path):
    p = tmp_path / "calib.jsonl"
    p.write_text(
        json.dumps({"text": "a" * 30}) + "\n" +
        json.dumps({"prompt": "b" * 30}) + "\n" +
        json.dumps({"text": "short"}) + "\n" +  # <20 chars, skipped
        "not-json-skipped\n" +
        json.dumps({"content": "c" * 30}) + "\n"
    )
    monkeypatch.setattr(common, "CALIB_PATH", str(p))
    out = common._load_calib_prompts()
    assert out is not None
    assert len(out) == 3


def test_eval_full_smoke_imports(monkeypatch):
    # eval_full imports eval_common; should load cleanly with no server.
    sys.path.insert(0, str(REPO / "scripts"))
    mod = _load("eval_full", "scripts/eval_full.py")
    # Cloud-matching default limits.
    assert mod.FULL_LIMITS == {
        "mmlu_pro": None, "ifeval": None,
        "gpqa_diamond_cot_zeroshot": 0.5,
    }


def test_eval_fast_smoke_imports():
    sys.path.insert(0, str(REPO / "scripts"))
    mod = _load("eval_fast", "scripts/eval_fast.py")
    # Fast iteration limits.
    assert mod.SMOKE_LIMITS == {
        "mmlu_pro": 0.10, "ifeval": 0.30,
        "gpqa_diamond_cot_zeroshot": 0.50,
    }
    # GPQA matches eval_full (cloud sample size — preserved for noise floor).
    full = _load("eval_full", "scripts/eval_full.py")
    assert mod.SMOKE_LIMITS["gpqa_diamond_cot_zeroshot"] == \
        full.FULL_LIMITS["gpqa_diamond_cot_zeroshot"]


def test_task_result_has_status_field(common):
    # status field is the cloud "—" mirror. Must default to nothing — caller
    # has to set it explicitly to either SCORED/TIMEOUT/ERROR.
    tr = common.TaskResult(
        name="x", score=0.5, threshold=0.4, status="SCORED",
        passed=True, limit=None, n_questions=10, wall_time_s=12.3,
        timeout_s=1500, thinking=False,
    )
    assert tr.status == "SCORED"
    assert tr.passed is True
    # subtask_scores defaults to empty (IFEval/GPQA case).
    assert tr.subtask_scores == {}


def test_task_result_can_carry_subtask_scores(common):
    tr = common.TaskResult(
        name="mmlu_pro", score=0.72, threshold=0.621, status="SCORED",
        passed=True, limit=None, n_questions=12032, wall_time_s=2400.0,
        timeout_s=3600, thinking=False,
        subtask_scores={"math": 0.65, "law": 0.81, "biology": 0.70},
    )
    assert tr.subtask_scores["math"] == 0.65
    assert len(tr.subtask_scores) == 3


def test_default_task_timeout_is_conservative(common):
    # 1500s (25 min) is the calibration anchor — looser would let cyankiwi-seq8's
    # MMLU-Pro slip past locally when cloud actually timed it out.
    assert common.DEFAULT_TASK_TIMEOUT_S == 1500


def test_fetch_mtp_counters_returns_none_when_unreachable(common):
    # /metrics scrape is best-effort — must fail silently rather than crash
    # the latency probe when the model is non-speculative or url is down.
    assert common._fetch_mtp_counters("http://127.0.0.1:1") is None


def test_latency_result_has_mtp_fields(common):
    lr = common.LatencyResult(realistic=True, runs=5, warmup=2)
    assert lr.mtp_accepted_length is None
    assert lr.mtp_accepted_total is None
    assert lr.mtp_drafts_total is None


def test_task_result_partial_fields_default_none(common):
    # Partial fields populate only on TIMEOUT (and only for LocalLM path).
    # Normal SCORED runs must leave them None so JSON stays small.
    tr = common.TaskResult(
        name="ifeval", score=0.82, threshold=0.814, status="SCORED",
        passed=True, limit=1.0, n_questions=541, wall_time_s=300.0,
        timeout_s=1500, thinking=False,
    )
    assert tr.partial_completed is None
    assert tr.partial_total is None


def test_task_result_can_carry_partial_on_timeout(common):
    tr = common.TaskResult(
        name="gpqa_diamond", score=None, threshold=0.630, status="TIMEOUT",
        passed=False, limit=0.5, n_questions=None, wall_time_s=1502.0,
        timeout_s=1500, thinking=True,
        partial_completed=42, partial_total=99,
    )
    assert tr.partial_completed == 42
    assert tr.partial_total == 99
    # Projected cloud score if remainder count as 0:
    # (42/99) * local_partial_score = something we can compute downstream


def test_eval_full_cli_parses(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    mod = _load("eval_full", "scripts/eval_full.py")
    # argparse smoke: required --model-name, defaults make sense.
    monkeypatch.setattr(sys, "argv",
                        ["eval_full.py", "--model-name", "x",
                         "--skip-latency", "--ping-timeout", "1",
                         "--container-url", "http://127.0.0.1:1"])
    # Don't actually run — but parser construction + ping fail returns 2.
    rc = mod.main()
    assert rc == 2


def test_eval_fast_cli_parses(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    mod = _load("eval_fast", "scripts/eval_fast.py")
    monkeypatch.setattr(sys, "argv",
                        ["eval_fast.py", "--model-name", "x",
                         "--skip-latency", "--ping-timeout", "1",
                         "--container-url", "http://127.0.0.1:1"])
    rc = mod.main()
    assert rc == 2
