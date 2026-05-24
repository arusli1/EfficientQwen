"""Shared eval primitives for scripts/eval_full.py and scripts/eval_fast.py.

Mirrors eval/run_quality_local.py (FROZEN per CLAUDE.md rule 4): same lm-eval
tasks, same num_fewshot, same thinking-mode rules, same chat-template, same
metric extraction. Two reasons to live in scripts/ rather than import eval/:

  1. eval/* must not be imported as a runtime dep from outside eval/ — keeps
     the "frozen contract" boundary clean.
  2. These wrappers add things eval/ deliberately omits: per-task wall-time,
     per-task limit overrides (cloud uses 100/100/50% — eval/ uses uniform),
     reusable Result/Run helpers for any model behind any CONTAINER_URL.

Source-of-truth params (mmlu_pro 5-shot no-think, ifeval 0-shot no-think,
gpqa_diamond_cot_zeroshot 0-shot thinking-ON; gates 0.621/0.814/0.630) MUST
match eval/run_quality_local.py — if that file ever changes (it shouldn't),
this needs the same edit.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

# ─── Cloud-matching constants (must equal eval/run_quality_local.py) ─────────
# Tuple shape: (lm_eval_task, result_key, num_fewshot, metric_key, gate, thinking)
QUALITY_TASKS: list[tuple[str, str, int, str, float, bool]] = [
    ("mmlu_pro",                  "mmlu_pro",     5, "exact_match,custom-extract",   0.621, False),
    ("ifeval",                    "ifeval",       0, "inst_level_strict_acc,none",   0.814, False),
    ("gpqa_diamond_cot_zeroshot", "gpqa_diamond", 0, "exact_match,flexible-extract", 0.630, True),
]

# Per-category latency baselines (ms) — arithmetic mean of per-category
# speedup is what the benchmark ranks on (docs/STATE.md §"Latency").
BASELINE_LATENCY_MS = {"short": 2582, "medium": 5441, "long": 6576}


# ─── HTTP plumbing ────────────────────────────────────────────────────────────
def _http_invoke(url: str, payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/invocations", data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def _invoke_completion(url: str, prompt: str, max_tokens: int) -> tuple[str, int]:
    # seed=1234 matches lm-eval's local-chat-completions client (cloud-parity:
    # `openai_completions.py:LocalChatCompletion._create_payload` injects this
    # → vLLM rng aligned with cloud's request stream for MTP-near-tie cases).
    r = _http_invoke(url, {"prompt": prompt, "max_tokens": max_tokens,
                           "temperature": 0.0, "seed": 1234})
    text = r.get("choices", [{}])[0].get("text", "")
    used = int(r.get("usage", {}).get("completion_tokens", 0) or 0)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return cleaned, used


def _invoke_chat(url: str, prompt: str, max_tokens: int,
                 thinking: bool = False) -> tuple[str, int]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
        # seed=1234 = lm-eval's TemplateAPI default (Expert C cloud-parity fix).
        "seed": 1234,
    }
    if thinking:
        payload["thinking"] = True
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    r = _http_invoke(url, payload, timeout=600)
    text = r.get("choices", [{}])[0].get("message", {}).get("content", "")
    used = int(r.get("usage", {}).get("completion_tokens", 0) or 0)
    return text, used


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% default).
    Returns (lo, hi). Same formula scipy uses; cheap, robust at low n
    where normal-approx CI is unstable."""
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    delta = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5) / denom
    return (max(0.0, center - delta), min(1.0, center + delta))


def wait_for_ping(url: str, timeout_s: int = 900) -> bool:
    """Poll {url}/ping until it returns 200 or deadline. Print every 30s."""
    deadline = time.time() + timeout_s
    start = time.time()
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"{url}/ping", timeout=2)
            if r.status == 200:
                print(f"[ping] ready after {time.time()-start:.0f}s", flush=True)
                return True
        except Exception:
            pass
        if int(time.time() - start) % 30 == 0:
            print(f"[ping] still waiting ({int(time.time()-start)}s)", flush=True)
        time.sleep(2)
    return False


# ─── lm-eval LM adapter — same shape as eval/run_quality_local.py:LocalLM ────
from lm_eval.api.model import LM as _LM


class _LocalLM(_LM):
    def __init__(self, url: str, thinking: bool, concurrency: int,
                 cancel_event: threading.Event | None = None):
        super().__init__()
        self.url = url
        self.thinking = thinking
        self.concurrency = concurrency
        # When set externally, every new request raises TimeoutError so
        # generate_until exits quickly. Used by run_task's watchdog to
        # implement hard mid-task pre-emption (matching cloud's "—" behavior:
        # exceed wall budget → no score, task aborts).
        self.cancel_event = cancel_event
        # Live counters — run_task reads these on TIMEOUT to record how much
        # got done before the watchdog fired. GIL makes int increment atomic
        # enough for this use (no consistency requirement across threads).
        self.completed_count = 0
        self.total_requests = 0
        # Truncation tracking: count requests whose completion_tokens == cap.
        # >5% = max_tokens budget is contributing to score loss.
        self.truncated_count = 0

    def generate_until(self, requests):
        total = len(requests)
        self.total_requests = total
        print(f"  [generate_until] {total} reqs "
              f"(thinking={self.thinking}, concurrency={self.concurrency})",
              flush=True)
        out: list[str | None] = [None] * total

        def _do(idx: int):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise TimeoutError("task budget exceeded (LocalLM cancelled)")
            context, gen_kwargs = requests[idx].args
            # Per-task cap matching cloud's behavior:
            #   GPQA thinking-on: 12288 (cloud setting per STATE.md)
            #   IFEval/MMLU-Pro: respect task YAML max_gen_toks if specified,
            #     bounded by a generous 2048 cap. IFEval yaml requests 1280;
            #     prior 512 default silently truncated long-form IFEval answers.
            default_max = 12288 if self.thinking else 2048
            # None-safe: lm-eval sometimes stores explicit max_gen_toks=None
            # for tasks; `min(None, default_max)` would TypeError-kill the
            # whole eval (stress-test finding HIGH).
            requested = (gen_kwargs.get("max_gen_toks") or
                         gen_kwargs.get("max_new_tokens") or default_max)
            max_tokens = min(requested, default_max)
            for attempt in range(3):
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise TimeoutError("task budget exceeded (LocalLM cancelled)")
                try:
                    if self.thinking:
                        text, used = _invoke_chat(self.url, context, max_tokens,
                                                  thinking=True)
                    else:
                        # The "ends with step-by-step" heuristic matches eval/.
                        tail = context.rstrip()[-30:]
                        if tail.endswith("step by step.") or "Answer: Let" in tail:
                            text, used = _invoke_completion(self.url, context,
                                                            max_tokens)
                        else:
                            text, used = _invoke_chat(self.url, context, max_tokens,
                                                      thinking=False)
                    if used >= max_tokens:
                        self.truncated_count += 1
                    return idx, text
                except TimeoutError:
                    raise
                except Exception:
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))
            return idx, ""

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = {ex.submit(_do, i): i for i in range(total)}
            for fut in as_completed(futs):
                idx, text = fut.result()
                out[idx] = text
                self.completed_count += 1
                if self.completed_count == 1:
                    print(f"  sample: [{text[:120]}]", flush=True)
                if (self.completed_count % max(1, total // 10) == 0
                        or self.completed_count == total):
                    elapsed = time.perf_counter() - t0
                    rate = self.completed_count / elapsed if elapsed > 0 else 0
                    eta = (total - self.completed_count) / rate if rate else 0
                    print(f"  {self.completed_count}/{total} | {rate:.2f} req/s | "
                          f"ETA {eta:.0f}s", flush=True)
        return [t or "" for t in out]

    # Required-but-unused interface — quality eval is generate-only.
    def loglikelihood(self, requests): return [(0.0, False)] * len(requests)
    def loglikelihood_rolling(self, requests): return [(0.0,)] * len(requests)
    @property
    def eot_token_id(self): return 0
    @property
    def max_length(self): return 16384
    @property
    def max_gen_toks(self): return 512
    @property
    def batch_size(self): return 1
    @property
    def device(self): return "cpu"
    def tok_encode(self, s): return list(s.encode())
    def tok_decode(self, t): return bytes(t).decode(errors="replace")
    def set_cache_hook(self, cache_hook): pass


# ─── MMLU-Pro path — needs a keep-alive proxy in front of the container ──────
# lm-eval's local-chat-completions client expects HTTP/1.1 keep-alive; our
# container's stdlib http.server doesn't speak it natively. The eval/ scripts
# install a tiny threading proxy that adds the right headers. Same trick here.
def _start_mmlu_proxy(target_url: str, port: int = 18080) -> HTTPServer:
    """Spawn the keep-alive proxy and return the server object so callers can
    `.shutdown()` it (used by run_task's watchdog to pre-empt MMLU-Pro on
    timeout — lm-eval's local-chat-completions client has no cancellation
    hook, so killing the proxy is the cleanest abort).

    NOTE: default port=18080 is hardcoded into the lm-eval `model_args`
    base_url in `run_task`. For PARALLEL eval runs (Expert D backlog), pass
    `port=0` here AND propagate `srv.server_address[1]` into the model_args
    URL. Today's runs are single-tenant so 18080 is fine."""
    class _Server(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def do_POST(self):
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                req = urllib.request.Request(
                    f"{target_url}/invocations", data=body,
                    headers={"Content-Type": "application/json"},
                )
                result = urllib.request.urlopen(req, timeout=120).read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(result)))
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.wfile.write(result)
                self.wfile.flush()
            except BrokenPipeError:
                pass
            except Exception as e:
                try:
                    err = json.dumps({"error": str(e)}).encode()
                    self.send_response(500)
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                except Exception:
                    pass

    srv = _Server(("127.0.0.1", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ─── Per-task runner ──────────────────────────────────────────────────────────
@dataclass
class TaskResult:
    name: str
    score: float | None
    threshold: float
    # status mirrors the target host's "—" semantics: TIMEOUT means the eval
    # wall time exceeded `timeout_s` — the target eval host kills the run and
    # reports "—" in the dashboard. SCORED means the task ran to completion
    # within budget. ERROR means simple_evaluate raised before producing any
    # score.
    status: str  # "SCORED" | "TIMEOUT" | "ERROR"
    passed: bool
    limit: float | None
    n_questions: int | None
    wall_time_s: float
    timeout_s: float | None
    thinking: bool
    # MMLU-Pro reports a per-subject score per subtask (`mmlu_pro_math`,
    # `mmlu_pro_law`, etc.). The headline `score` field is the simple mean.
    # When a candidate regresses, the per-subject map is what lets you bisect
    # WHICH subject broke (e.g. math collapse vs uniform drop). Empty dict
    # for IFEval/GPQA (single-task) — keeps schema uniform.
    subtask_scores: dict = field(default_factory=dict)
    # Partial completion info: when watchdog fires (TIMEOUT), how many
    # requests had completed before abort. Lets you project cloud's truncated
    # score: if N/total finished at local_score X, cloud would score
    # ~ (N × X) / total (rest count as empty/0). Only populated for the
    # LocalLM path (IFEval/GPQA) — MMLU-Pro client has no equivalent hook.
    partial_completed: int | None = None
    partial_total: int | None = None
    # Wilson 95% CI on `score`. For IFEval/GPQA where score is a per-sample
    # accuracy proportion, this captures sample-size noise. The ship-rule
    # "ci_lower > threshold + 0.01" is the variance-aware gate: only ship a
    # candidate whose local CI lower bound clears the gate by ≥1pp.
    # MMLU-Pro: still computed but is approximate (n is sum across subjects).
    ci_lower: float | None = None
    ci_upper: float | None = None
    # Truncation diagnostic: fraction of LocalLM requests whose
    # completion_tokens hit max_tokens (the cap). >5% means the budget cuts
    # off reasoning and lowers score artificially. Only LocalLM path
    # (IFEval/GPQA) — MMLU-Pro uses lm-eval's own client and doesn't expose.
    truncation_count: int | None = None
    truncation_rate: float | None = None


# Default per-task wall-time budget in seconds. The cyankiwi-seq8 MMLU-Pro run
# is the anchor data point: a prior cloud trace timed it out, so cloud's hidden
# budget is somewhere south of cyankiwi-seq8 MMLU-Pro's actual local wall time.
# 1500s (25 min) is conservative; user can raise via --task-timeout when
# probing slower configurations or lower when stress-testing leaner ones.
DEFAULT_TASK_TIMEOUT_S = 1500


def run_task(url: str, task: tuple, limit: float | None,
             concurrency: int = 8,
             timeout_s: float | None = DEFAULT_TASK_TIMEOUT_S,
             samples_jsonl_path: str | None = None) -> TaskResult:
    """Run one lm-eval task against `url`. `limit` ∈ (0,1] | None (=full).

    Post-hoc TIMEOUT detection: we record actual wall time, and if it exceeds
    `timeout_s`, the result is marked TIMEOUT (passed=False, score retained
    for diagnostic value). This predicts cloud "—" without paying mid-task
    kill complexity — the existing run still completes so we see how far over
    budget we were. Set `timeout_s=None` to disable.
    """
    from lm_eval import simple_evaluate

    task_name, result_key, num_fewshot, metric_key, threshold, thinking = task
    print(f"\n[{result_key}] {task_name} (n-fewshot={num_fewshot}, "
          f"thinking={thinking}, limit={limit}, timeout={timeout_s}s)",
          flush=True)
    t0 = time.perf_counter()

    status = "SCORED"
    score: float | None = None
    n_qs: int | None = None
    eval_out: dict = {}

    # Hard pre-emption: watchdog fires at timeout_s and cancels the task.
    # For LocalLM path (IFEval/GPQA): sets cancel_event → next request raises
    # TimeoutError → simple_evaluate exits with the partial-results exception.
    # For MMLU-Pro path: shuts down the proxy server → in-flight + new
    # requests fail → simple_evaluate exits. The watchdog logs that it fired
    # so we can distinguish TIMEOUT from script errors.
    cancel_event = threading.Event()
    watchdog_fired = threading.Event()
    proxy_srv: HTTPServer | None = None

    def _watchdog():
        if timeout_s is None:
            return
        if not cancel_event.wait(timeout=timeout_s):
            # Reached timeout without main completing — fire pre-emption.
            watchdog_fired.set()
            cancel_event.set()
            print(f"  [{result_key}] WATCHDOG firing at {timeout_s}s — "
                  f"aborting task", flush=True)
            if proxy_srv is not None:
                try:
                    proxy_srv.shutdown()
                except Exception:
                    pass

    watch_thread = threading.Thread(target=_watchdog, daemon=True)
    watch_thread.start()

    local_lm: _LocalLM | None = None
    try:
        if task_name == "mmlu_pro":
            proxy_srv = _start_mmlu_proxy(url, port=18080)
            eval_out = simple_evaluate(
                model="local-chat-completions",
                model_args=(f"model=Qwen/Qwen3.5-4B,"
                            f"base_url=http://localhost:18080/v1/chat/completions,"
                            f"tokenized_requests=False,num_concurrent={concurrency},"
                            f"eos_string=<|im_end|>,timeout=120"),
                tasks=[task_name], num_fewshot=num_fewshot, batch_size=1,
                limit=limit, apply_chat_template=True,
                # Cloud runs lm-eval 0.4.11 where fewshot_as_multiturn
                # defaults False; local 0.4.12 defaults True. Pin to False
                # for cloud parity on MMLU-Pro 5-shot.
                fewshot_as_multiturn=False,
                # Cache tokenized requests across re-runs against the same
                # model — saves ~20-60s startup on MMLU-Pro's 12k docs.
                cache_requests=True,
                random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                fewshot_random_seed=0,
                confirm_run_unsafe_code=True,
            )
        else:
            local_lm = _LocalLM(url, thinking=thinking, concurrency=concurrency,
                                cancel_event=cancel_event)
            eval_out = simple_evaluate(
                model=local_lm,
                tasks=[task_name], num_fewshot=num_fewshot, batch_size=1,
                limit=limit, random_seed=0, numpy_random_seed=1234,
                torch_random_seed=1234, confirm_run_unsafe_code=True,
            )
    except Exception as e:
        if watchdog_fired.is_set():
            print(f"  [{result_key}] timed out cleanly mid-task", flush=True)
        else:
            status = "ERROR"
            print(f"  [{result_key}] ERROR mid-eval: {e}", flush=True)
    finally:
        # Signal watchdog to exit if it hasn't fired (normal completion path)
        cancel_event.set()
        if proxy_srv is not None:
            try:
                proxy_srv.shutdown()
            except Exception:
                pass

    wall = time.perf_counter() - t0
    task_results = eval_out.get("results", {})
    samples_block = eval_out.get("samples", {})
    if isinstance(samples_block, dict):
        # n_qs = count of UNIQUE doc_ids across the samples block. lm-eval
        # emits one sample row per (doc × filter), so a 99-Q GPQA task with
        # 2 filters (strict-match + flexible-extract) has 198 sample rows for
        # 99 unique questions. The Wilson CI denominator must be the unique
        # count (independent observations), not row count, or CI is too
        # narrow by √2 (subagent stress-test finding, severity HIGH).
        unique_doc_ids = set()
        for subtask_name, subtask_samples in samples_block.items():
            if not isinstance(subtask_samples, list):
                continue
            for s in subtask_samples:
                if isinstance(s, dict) and "doc_id" in s:
                    # Compose subtask+doc_id so MMLU-Pro's per-subtask
                    # numbering doesn't collide (subtask A doc 0 ≠ subtask B doc 0).
                    unique_doc_ids.add((subtask_name, s["doc_id"]))
        n_qs = len(unique_doc_ids) if unique_doc_ids else None

    # Optional per-sample JSONL dump for post-hoc bisects (first-N vs rest,
    # by-category, by-difficulty). Slim payload — full thinking traces would
    # be megabytes per sample; we save what's needed for score recomputation.
    if samples_jsonl_path and isinstance(samples_block, dict):
        os.makedirs(os.path.dirname(samples_jsonl_path) or ".", exist_ok=True)
        with open(samples_jsonl_path, "w") as f:
            for subtask, samples in samples_block.items():
                if not isinstance(samples, list):
                    continue
                for s in samples:
                    if not isinstance(s, dict):
                        continue
                    slim = {
                        "subtask": subtask,
                        "doc_id": s.get("doc_id"),
                        "target": s.get("target"),
                        "filtered_resps": s.get("filtered_resps"),
                        # Raw model response — needed for postprocess analysis
                        # (B5 hack: parse for intended answer when extraction
                        # picked wrong letter). Each can be 10-40KB for
                        # thinking-mode GPQA; full GPQA is a few MB.
                        "resps": s.get("resps"),
                    }
                    # Copy any metric values (exact_match, etc.) the task emits
                    for k, v in s.items():
                        if k in {"doc", "arguments", "resps", "filtered_resps",
                                 "doc_id", "target"}:
                            continue
                        if isinstance(v, (int, float, str, bool, type(None))):
                            slim[k] = v
                    f.write(json.dumps(slim, default=str) + "\n")
        print(f"  [{result_key}] wrote {n_qs} samples to {samples_jsonl_path}",
              flush=True)

    subtask_scores: dict = {}
    if task_name == "mmlu_pro":
        # mmlu_pro splits into per-subject subtasks (mmlu_pro_math, ...).
        # Headline `score` = mean over subjects. Per-subject map preserved
        # so debugging "which subject regressed" is one JSON field away.
        subtask_scores = {
            k.removeprefix("mmlu_pro_"): round(float(v[metric_key]), 4)
            for k, v in task_results.items()
            if k.startswith("mmlu_pro_") and isinstance(v, dict)
            and v.get(metric_key) is not None
        }
        score = (sum(subtask_scores.values()) / len(subtask_scores)
                 if subtask_scores else None)
    else:
        single = task_results.get(task_name, {})
        score = single.get(metric_key)
        if score is None:
            base = metric_key.split(",")[0]
            for k, v in single.items():
                if base in k and isinstance(v, (int, float)):
                    score = v
                    break

    score = round(float(score), 4) if score is not None else None

    # Watchdog fired → TIMEOUT (with whatever partial score lm-eval salvaged,
    # often None). Post-hoc wall check is the belt-and-suspenders for the
    # case where simple_evaluate completed but ran over budget (e.g. tight
    # smoke timeout that wasn't quite enough to abort cleanly).
    if watchdog_fired.is_set():
        status = "TIMEOUT"
    elif status == "SCORED" and timeout_s is not None and wall > timeout_s:
        status = "TIMEOUT"

    passed = status == "SCORED" and score is not None and score >= threshold
    badge = {"SCORED": "PASS" if passed else "FAIL",
             "TIMEOUT": "TIMEOUT",
             "ERROR": "ERROR"}[status]
    print(f"  [{result_key}] {badge} | score={score} gate={threshold} "
          f"| wall={wall:.1f}s budget={timeout_s}s | n={n_qs}", flush=True)
    partial_completed = (local_lm.completed_count
                         if local_lm and status == "TIMEOUT" else None)
    partial_total = (local_lm.total_requests
                     if local_lm and status == "TIMEOUT" else None)
    ci_lower = ci_upper = None
    if score is not None and n_qs and n_qs > 0:
        lo, hi = wilson_ci(score, n_qs)
        ci_lower, ci_upper = round(lo, 4), round(hi, 4)
    truncation_count = truncation_rate = None
    if local_lm is not None and local_lm.total_requests > 0:
        truncation_count = local_lm.truncated_count
        # Denominator is total_requests, not completed_count: on TIMEOUT
        # the watchdog kills mid-batch, so completed_count understates total
        # and inflates the rate. total_requests is the truth (Expert B fix).
        truncation_rate = round(
            local_lm.truncated_count / local_lm.total_requests, 4
        )
    return TaskResult(
        name=result_key, score=score, threshold=threshold, status=status,
        passed=passed, limit=limit, n_questions=n_qs,
        wall_time_s=round(wall, 2),
        timeout_s=timeout_s, thinking=thinking,
        subtask_scores=subtask_scores,
        partial_completed=partial_completed, partial_total=partial_total,
        ci_lower=ci_lower, ci_upper=ci_upper,
        truncation_count=truncation_count, truncation_rate=truncation_rate,
    )


# ─── Latency probe — realistic prompts when available, FILLER fallback ───────
# We prefer the calibration set used by scripts/bench_latency.py because the
# target host uses a private test set of natural prompts. FILLER ("quick brown
# fox" repeated) overstates MTP acceptance and prefix-cache hit rate. Falls back
# to FILLER if the calib corpus is missing, logging the substitution so the
# result JSON is self-documenting.
FILLER = "The quick brown fox jumps over the lazy dog. "
PROMPT_CONFIGS = {
    "short":  {"num_tokens": 64,   "max_new_tokens": 128},
    "medium": {"num_tokens": 2048, "max_new_tokens": 256},
    "long":   {"num_tokens": 8192, "max_new_tokens": 256},
}
CALIB_PATH = "data/calibration_1024.jsonl"


def _load_calib_prompts() -> list[str] | None:
    if not os.path.isfile(CALIB_PATH):
        return None
    out: list[str] = []
    with open(CALIB_PATH) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("prompt") or obj.get("content") or ""
            if text and len(text.strip()) >= 20:
                out.append(text)
    return out or None


def _shape(text: str, tokens: int) -> str:
    chars = tokens * 4  # ~4 chars/token for Qwen3.5 BPE
    if len(text) < chars:
        sep = "\n\n---\n\n"
        rep = (text + sep) * (chars // (len(text) + len(sep)) + 1)
        return rep[:chars]
    return text[:chars]


@dataclass
class LatencyResult:
    realistic: bool
    runs: int
    warmup: int
    per_category: dict = field(default_factory=dict)  # cat -> {median_ms, min_ms, max_ms, speedup}
    avg_speedup: float | None = None
    # MTP acceptance length captured around the latency probe (delta of
    # vLLM's prometheus counters between probe start and end). MTP=K means
    # max acceptance length = K+1. cyankiwi-seq8 at the K-sweep optimum K*=4
    # typically lands ~3-4 on natural prompts. Drops here = invisible speed
    # killer. Older configs ran MTP=7; K=4 is the sweep optimum.
    mtp_accepted_length: float | None = None
    mtp_accepted_total: int | None = None
    mtp_drafts_total: int | None = None


def _fetch_mtp_counters(url: str) -> tuple[int, int] | None:
    """Returns (accepted_tokens, drafts) deltas baseline from /metrics, or None
    if the endpoint isn't reachable / model is non-speculative."""
    try:
        body = urllib.request.urlopen(f"{url}/metrics", timeout=5).read().decode()
    except Exception:
        return None
    # Sum across all engines (multi-engine TP>1 exposes per-engine labels;
    # OVERWRITE bug previously dropped all but the last engine's count).
    accepted = drafts = 0
    accepted_seen = drafts_seen = False
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Lines: 'vllm:spec_decode_num_accepted_tokens_total{...} 572740.0'
        if "vllm:spec_decode_num_accepted_tokens_total" in line and "{" in line:
            try:
                accepted += int(float(line.rsplit(" ", 1)[1]))
                accepted_seen = True
            except ValueError:
                pass
        elif "vllm:spec_decode_num_drafts_total" in line and "{" in line:
            try:
                drafts += int(float(line.rsplit(" ", 1)[1]))
                drafts_seen = True
            except ValueError:
                pass
    if not (accepted_seen and drafts_seen):
        return None
    return accepted, drafts


def run_latency(url: str, runs: int = 5, warmup: int = 2,
                seed: int = 0) -> LatencyResult:
    import random
    rng = random.Random(seed)
    calib = _load_calib_prompts()
    realistic = calib is not None
    if not realistic:
        print(f"[latency] {CALIB_PATH} missing — using FILLER (inflated)",
              flush=True)

    res = LatencyResult(realistic=realistic, runs=runs, warmup=warmup)
    mtp_before = _fetch_mtp_counters(url)
    for cat, cfg in PROMPT_CONFIGS.items():
        prompts: list[str] = []
        if realistic:
            for _ in range(warmup + runs):
                prompts.append(_shape(rng.choice(calib), cfg["num_tokens"]))
        else:
            base = FILLER * max(1, cfg["num_tokens"] // 10)
            prompts = [base] * (warmup + runs)

        # Warmup (untimed)
        for p in prompts[:warmup]:
            try:
                _ = _invoke_completion(url, p, cfg["max_new_tokens"])
            except Exception:
                pass

        times_ms = []
        for i, p in enumerate(prompts[warmup:]):
            t0 = time.perf_counter()
            try:
                _ = _invoke_completion(url, p, cfg["max_new_tokens"])
            except Exception as e:
                print(f"[latency] {cat} run {i+1} ERR: {e}", flush=True)
                continue
            ms = (time.perf_counter() - t0) * 1000
            times_ms.append(ms)
            print(f"  [{cat}] run {i+1}/{runs}: {ms:.1f}ms", flush=True)

        if not times_ms:
            res.per_category[cat] = {"error": "no successful runs"}
            continue
        median = round(statistics.median(times_ms), 2)
        speedup = round(BASELINE_LATENCY_MS[cat] / median, 3)
        # quantiles: only meaningful when we have ≥4 samples; for small `runs`
        # (e.g. eval_fast default 3) fall back to median for both p50/p95.
        if len(times_ms) >= 4:
            qs = statistics.quantiles(times_ms, n=20)  # 5%-step quantiles
            p50, p95 = round(qs[9], 2), round(qs[18], 2)
        else:
            p50 = median
            p95 = round(max(times_ms), 2)
        res.per_category[cat] = {
            "median_ms": median,
            "p50_ms": p50,
            "p95_ms": p95,
            "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2),
            "baseline_ms": BASELINE_LATENCY_MS[cat],
            "speedup": speedup,
        }
        print(f"  [{cat}] median={median}ms p95={p95}ms speedup={speedup}×",
              flush=True)

    speedups = [v["speedup"] for v in res.per_category.values()
                if "speedup" in v]
    if speedups:
        res.avg_speedup = round(sum(speedups) / len(speedups), 3)
        print(f"\n[latency] avg-of-per-cat speedup: {res.avg_speedup}× "
              f"(realistic={realistic})", flush=True)
    mtp_after = _fetch_mtp_counters(url)
    if mtp_before is not None and mtp_after is not None:
        accepted_delta = mtp_after[0] - mtp_before[0]
        drafts_delta = mtp_after[1] - mtp_before[1]
        if drafts_delta > 0:
            # acceptance length = 1 base token + accepted_drafts / drafts_count
            res.mtp_accepted_length = round(1 + accepted_delta / drafts_delta, 3)
            res.mtp_accepted_total = accepted_delta
            res.mtp_drafts_total = drafts_delta
            print(f"[latency] MTP acceptance length: "
                  f"{res.mtp_accepted_length} "
                  f"({accepted_delta} accepted / {drafts_delta} drafts)",
                  flush=True)
    return res


# ─── Aggregation + verdict ────────────────────────────────────────────────────
@dataclass
class EvalRun:
    mode: str                          # "full" | "fast"
    model_name: str
    container_url: str
    started_utc: str
    wall_total_s: float
    quality: list[TaskResult]
    latency: LatencyResult | None
    all_gates_passed: bool
    notes: dict = field(default_factory=dict)


def _git_sha() -> str:
    """Returns short HEAD sha (or 'unknown') without subprocess.run overhead in
    the common case where .git/HEAD points at a ref. Read-only filesystem walk.
    Anchored on repo root (parent of scripts/) so it works regardless of CWD."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(repo_root, ".git/HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            with open(os.path.join(repo_root, ".git", head[5:])) as f:
                return f.read().strip()[:8]
        return head[:8]
    except Exception:
        return "unknown"


def write_result(run: EvalRun, out_path: str) -> None:
    """Write result to disk ATOMICALLY (tmp + os.replace) so a crashed parent
    can't leave a half-written JSON. Includes schema_version + git_sha + host
    for cross-cycle traceability (Expert D backlog)."""
    import socket
    payload = {
        "schema_version": 2,
        "mode": run.mode,
        "model_name": run.model_name,
        "container_url": run.container_url,
        "started_utc": run.started_utc,
        "wall_total_s": round(run.wall_total_s, 1),
        "all_gates_passed": run.all_gates_passed,
        "quality": [asdict(t) for t in run.quality],
        "latency": asdict(run.latency) if run.latency else None,
        "notes": run.notes,
        # Provenance — pin to the code/env that produced this number.
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, out_path)
    print(f"\n[done] wrote {out_path}", flush=True)


def print_verdict(run: EvalRun) -> None:
    print("\n" + "=" * 70)
    print(f"VERDICT — {run.mode.upper()} eval, model={run.model_name}")
    print("=" * 70)
    print(f"  wall total: {run.wall_total_s/60:.1f} min")
    for t in run.quality:
        if t.status == "TIMEOUT":
            partial = ""
            if t.partial_completed is not None and t.partial_total:
                pct = 100 * t.partial_completed / t.partial_total
                partial = (f" — {t.partial_completed}/{t.partial_total} "
                           f"({pct:.0f}%) before kill")
            marker, score_str = "⏱", (
                f"{t.score} (TIMEOUT — cloud would show '—'{partial})"
            )
        elif t.status == "ERROR":
            marker, score_str = "✗", f"ERROR — score={t.score}"
        else:
            marker, score_str = ("✓" if t.passed else "✗"), str(t.score)
        sample = "FULL" if t.limit is None else f"{int(t.limit*100)}%"
        print(f"  {marker} {t.name:<14} score={score_str} gate={t.threshold} "
              f"(sample={sample}, n={t.n_questions}, "
              f"{t.wall_time_s:.0f}s/budget={t.timeout_s}s, "
              f"thinking={t.thinking})")
        if t.ci_lower is not None and t.ci_upper is not None:
            margin = t.ci_lower - t.threshold
            ship = "SHIP-SAFE" if margin >= 0.01 else (
                "SHIP-RISK" if margin >= 0.0 else "SHIP-FAIL"
            )
            print(f"      CI95=[{t.ci_lower:.4f}, {t.ci_upper:.4f}] "
                  f"ci_lower-gate={margin:+.4f} → {ship}")
        if t.truncation_rate is not None and t.truncation_count is not None:
            warn = " ⚠️ >5%" if t.truncation_rate > 0.05 else ""
            print(f"      truncated: {t.truncation_count}/{t.partial_total or t.n_questions or '?'} "
                  f"({t.truncation_rate*100:.1f}%) at max_tokens cap{warn}")
        # MMLU-Pro: print subject min/max to spot a single-subject collapse
        # at a glance. Full per-subject map lives in the JSON.
        if t.subtask_scores:
            sorted_s = sorted(t.subtask_scores.items(), key=lambda kv: kv[1])
            lo_k, lo_v = sorted_s[0]
            hi_k, hi_v = sorted_s[-1]
            print(f"      subjects: {len(t.subtask_scores)} · "
                  f"low={lo_k}={lo_v} · high={hi_k}={hi_v}")
    if run.latency:
        for cat, v in run.latency.per_category.items():
            if "median_ms" in v:
                print(f"  · {cat:<6} med={v['median_ms']:>7.0f}ms "
                      f"p95={v.get('p95_ms', v['median_ms']):>7.0f}ms "
                      f"({v['speedup']:.2f}×)")
        if run.latency.avg_speedup:
            tag = "realistic" if run.latency.realistic else "FILLER (inflated)"
            print(f"  · avg-of-per-cat speedup: "
                  f"{run.latency.avg_speedup:.2f}× ({tag})")
        if run.latency.mtp_accepted_length is not None:
            print(f"  · MTP acceptance length: "
                  f"{run.latency.mtp_accepted_length} "
                  f"(K+1=8 is max for MTP=7)")
    print(f"  ALL GATES: {'PASS' if run.all_gates_passed else 'FAIL'}")
    # Ship-safety: only SHIP-SAFE when EVERY quality task's CI lower bound
    # is ≥1pp above its gate. Single-task variance can fake a pass; this
    # rule requires the noise floor itself to clear the floor.
    ship_decisions = []
    for t in run.quality:
        if t.ci_lower is None or t.status != "SCORED":
            ship_decisions.append((t.name, "UNKNOWN"))
            continue
        margin = t.ci_lower - t.threshold
        if margin >= 0.01:
            ship_decisions.append((t.name, "SAFE"))
        elif margin >= 0.0:
            ship_decisions.append((t.name, "RISK"))
        else:
            ship_decisions.append((t.name, "FAIL"))
    order = ("FAIL", "UNKNOWN", "RISK", "SAFE")  # ascending safety
    worst = min((order.index(d[1]) for d in ship_decisions), default=0)
    overall = order[worst]
    detail = ", ".join(f"{n}={v}" for n, v in ship_decisions)
    print(f"  SHIP-GATE: {overall} ({detail})")


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
