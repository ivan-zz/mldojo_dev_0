"""Shared utilities for MLE-STAR subgraphs.

Provides tracing, checkpointing, structured logging, and utility functions
used across all subgraphs and algorithms.

Logging architecture:
    - MLELogger: custom logger that writes JSON lines to per-run log files
      and to stdout via Python logging handlers
    - Dev logger: rotating file handler on the mle_star root logger that
      always writes to runs/_dev/dev.log (10MB max, 3 backups)
    - ContextVars (_current_run_id, _current_run_dir, _current_phase)
      propagate run context through LangGraph fan-out tasks
    - Run directory format: runs/YYYYMMDDHHMMSS_{uuid8}/
    - Per-run log file: runs/{run_dir}/run.log (one JSON dict per line)
    - Dev log file: runs/_dev/dev.log (always active, rotating 10MB x 3)
    - INFO level → per-run file + stdout; DEBUG level → per-run file only
    - Dev log captures ALL events at DEBUG level regardless of MLELogger

Tracing architecture (Langfuse v4):
    - One root observation per pipeline run (created in run())
    - SubgraphSpan creates child observations under the root
    - @traceable nodes create nested observations under the SubgraphSpan parent
    - All observations share a single trace_id, producing a grouped flow in Langfuse

Session support:
    - propagate_attributes(session_id=...) sets the session ID in the OTel context
    - All observations created within the context inherit the session_id
    - Multiple pipeline runs with the same session_id are grouped in Langfuse Sessions view
    - Propagation works through LangGraph's copy_context() for fan-out tasks
"""

import json
import os
import re
import sys
import time
import uuid
import logging
import operator
import functools
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, TypedDict, Optional

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse, propagate_attributes

langfuse = Langfuse()

_FLUSH_PERIOD_STR = os.environ.get("LANGFUSE_FLUSH_PERIOD", "")
_FLUSH_PERIOD: float | None = None
if _FLUSH_PERIOD_STR == "0":
    _FLUSH_PERIOD = 0.0
elif _FLUSH_PERIOD_STR:
    try:
        _FLUSH_PERIOD = float(_FLUSH_PERIOD_STR)
    except ValueError:
        _FLUSH_PERIOD = None
_last_flush_time: float = 0.0

_logger = logging.getLogger("mle_star")


def _should_flush() -> bool:
    """Determine whether langfuse.flush() should be called now.

    Controlled by LANGFUSE_FLUSH_PERIOD environment variable:
        - Not set or empty: Never flush from traceable (root-only flush at run end).
        - "0": Flush after every observation (per-node flushing).
        - Positive float (e.g. "30"): Flush if this many seconds have elapsed
          since the last flush.

    Thread-safe for single-threaded LangGraph execution.
    """
    global _last_flush_time
    if _FLUSH_PERIOD is None:
        return False
    if _FLUSH_PERIOD == 0.0:
        return True
    now = time.time()
    if now - _last_flush_time >= _FLUSH_PERIOD:
        _last_flush_time = now
        return True
    return False


_current_obs: ContextVar[Optional[Any]] = ContextVar("_current_obs", default=None)
_current_run_id: ContextVar[Optional[str]] = ContextVar("_current_run_id", default=None)
_current_run_dir: ContextVar[Optional[str]] = ContextVar(
    "_current_run_dir", default=None
)
_current_phase: ContextVar[Optional[str]] = ContextVar("_current_phase", default=None)

MAX_FIX_RETRIES = 5

# ── LLM call counter (Stage 10: MAX_LLM_CALLS_PER_PHASE enforcement) ──────

_llm_call_count: ContextVar[int] = ContextVar("_llm_call_count", default=0)


def reset_llm_call_counter() -> None:
    """Reset the per-phase LLM call counter to zero."""
    _llm_call_count.set(0)


def get_llm_call_count() -> int:
    """Return the current LLM call count for this phase."""
    return _llm_call_count.get(0)


def _increment_and_check_llm_limit() -> None:
    """Increment LLM call counter and raise RuntimeError if limit exceeded."""
    from src.mle_star.config import MAX_LLM_CALLS_PER_PHASE

    current = _llm_call_count.get(0) + 1
    _llm_call_count.set(current)
    if current > MAX_LLM_CALLS_PER_PHASE:
        log_node_event(
            "call_llm",
            "limit_exceeded",
            {"count": current, "limit": MAX_LLM_CALLS_PER_PHASE},
        )
        raise RuntimeError(
            f"LLM call limit exceeded: {current} calls > {MAX_LLM_CALLS_PER_PHASE} limit. "
            f"Set MLE_MAX_LLM_CALLS_PER_PHASE to increase."
        )


# ── Per-node timeout (Stage 10: PER_NODE_TIMEOUT_SECONDS enforcement) ───────

import signal


def _timeout_handler(signum, frame):
    """Signal handler for SIGALRM — raises TimeoutError."""
    raise TimeoutError(
        f"Node execution timed out after {signum} seconds. "
        f"Set MLE_PER_NODE_TIMEOUT_SECONDS to increase."
    )


def run_with_timeout(func, timeout_seconds: int | None = None):
    """Run a function with a timeout using signal.alarm (POSIX only).

    Args:
        func: Callable to execute.
        timeout_seconds: Maximum execution time in seconds. If None, uses
            PER_NODE_TIMEOUT_SECONDS from config.

    Returns:
        The result of func().

    Raises:
        TimeoutError: If func() exceeds the timeout.
        RuntimeError: If signal.alarm is unavailable (non-POSIX, main thread).
    """
    from src.mle_star.config import PER_NODE_TIMEOUT_SECONDS

    if timeout_seconds is None:
        timeout_seconds = PER_NODE_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return func()

    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    except (ValueError, OSError):
        return func()

    try:
        result = func()
    finally:
        signal.alarm(0)
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, OSError):
            pass

    return result


_RUNS_ROOT = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
    "runs",
)

_DEV_LOG_DIR = os.path.join(_RUNS_ROOT, "_dev")
_DEV_LOG_PATH = os.path.join(_DEV_LOG_DIR, "dev.log")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "mle_json_entry"):
            return record.mle_json_entry
        return record.getMessage()


def _setup_dev_logger():
    """Set up the dev rotating file handler on the mle_star logger.

    Always writes to runs/_dev/dev.log with 10MB max, 3 backups.
    Captures all events at DEBUG level regardless of whether MLELogger is active.
    Idempotent: only adds the handler once even if called multiple times.
    """
    for h in _logger.handlers:
        if (
            isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == _DEV_LOG_PATH
        ):
            return h
    os.makedirs(_DEV_LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(
        _DEV_LOG_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_JsonFormatter())
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)
    return handler


_dev_handler = _setup_dev_logger()


class MLELogger:
    """Per-run structured JSON logger.

    Each pipeline run creates a run directory under runs/ with a run.log file.
    JSON lines are written both to the log file and to stdout via Python logging.

    ContextVars propagate run_id, run_dir, and phase through LangGraph's
    copy_context() for fan-out tasks, so log entries automatically include
    the correct run context without explicit parameter passing.

    Usage:
        mle_log = MLELogger(run_dir="/path/to/runs/20260514103000_abc123f")
        mle_log.info("A1__retrieve", "output", {"retrieved_models": ["LGBM"]})
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.run_id = os.path.basename(run_dir.rstrip("/"))
        os.makedirs(run_dir, exist_ok=True)

        self._log_path = os.path.join(run_dir, "run.log")

        self._logger = logging.getLogger(f"mle_star.{self.run_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if not self._logger.handlers:
            file_handler = logging.FileHandler(
                self._log_path, mode="a", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(file_handler)

            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.INFO)
            stdout_handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(stdout_handler)

    def _emit(self, level: int, entry: Dict[str, Any]):
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg="",
            args=(),
            exc_info=None,
        )
        record.mle_json_entry = json.dumps(entry, default=str, ensure_ascii=False)
        self._logger.handle(record)

    def log(
        self,
        node: str,
        event: str,
        data: Dict[str, Any] | None = None,
        duration_ms: float | None = None,
        run_id: str | None = None,
        phase: str | None = None,
        level: int = logging.INFO,
    ) -> None:
        effective_run_id = run_id or _current_run_id.get()
        effective_phase = phase or _current_phase.get()
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": effective_run_id,
            "node": node,
            "event": event,
            "phase": effective_phase,
            "data": data or {},
        }
        if duration_ms is not None:
            entry["duration_ms"] = round(duration_ms, 2)
        self._emit(level, entry)

    def debug(
        self, node: str, event: str, data: Dict[str, Any] | None = None, **kwargs
    ) -> None:
        self.log(node, event, data, level=logging.DEBUG, **kwargs)

    def info(
        self, node: str, event: str, data: Dict[str, Any] | None = None, **kwargs
    ) -> None:
        self.log(node, event, data, level=logging.INFO, **kwargs)

    def close(self) -> None:
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)


_current_mle_logger: ContextVar[Optional["MLELogger"]] = ContextVar(
    "_current_mle_logger", default=None
)


def get_mle_logger() -> MLELogger | None:
    return _current_mle_logger.get()


from langgraph.checkpoint.memory import InMemorySaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False

_SQLITE_RUNTIME_COMPATIBLE = False
if _SQLITE_AVAILABLE:
    try:
        import sqlite3
        from langgraph.graph import StateGraph, START, END
        from typing import Annotated, TypedDict
        import operator

        class _CompatState(TypedDict):
            v: int

        def _compat_node(state):
            return {"v": state["v"] + 1}

        _compat_builder = StateGraph(_CompatState)
        _compat_builder.add_node("n", _compat_node)
        _compat_builder.add_edge(START, "n")
        _compat_builder.add_edge("n", END)

        _compat_conn = sqlite3.connect(":memory:", check_same_thread=False)
        _compat_saver = SqliteSaver(_compat_conn)
        _compat_saver.setup()
        _compat_graph = _compat_builder.compile(checkpointer=_compat_saver)
        _compat_graph.invoke({"v": 0}, {"configurable": {"thread_id": "__compat__"}})
        _compat_conn.close()
        _SQLITE_RUNTIME_COMPATIBLE = True
    except Exception:
        _SQLITE_RUNTIME_COMPATIBLE = False


_TRACE_EXCLUDE_KEYS = {"stage_history", "sub_events"}

_NODE_INPUT_FIELDS = {
    "A1__retrieve": ["task_desc"],
    "A2__generate": ["model"],
    "A3__merge": ["base_code", "ref_code"],
    "A13__check_usage": ["model", "code"],
    "A13__fix_usage": ["model", "code", "status"],
    "A12__check_leakage": ["model", "code"],
    "A12__fix_leakage": ["model", "code", "status"],
    "eval_candidate": ["model", "code", "attempts"],
    "A11__debug": ["model", "code", "attempts", "status"],
    "A11__debug_merge": ["merged_code", "attempts", "status"],
    "A12__check_leakage_merge": ["merged_code"],
    "A12__fix_leakage_merge": ["merged_code", "status"],
    "eval_merge": ["merged_code", "attempts"],
    "generate_candidates": ["retrieved_models"],
    "candidate_flow": ["model"],
    "Rank": ["candidates_pool"],
    "merge_candidates": ["best_candidate", "leaderboard"],
    "merge_flow": ["base_code", "ref_code"],
    "SelectBest": ["candidates_pool"],
    "A4__generate_ablation": ["current_solution", "best_score"],
    "ablation_variant_flow": ["variant_name", "variant_code", "block_name"],
    "eval_ablation_variant": ["variant_name", "variant_code", "block_name", "attempts"],
    "A11__debug_ablation_variant": ["variant_code", "attempts", "execution_error"],
    "A5__summarize_ablation": ["ablation_results_list"],
    "A6__extract_block": [
        "ablation_results_list",
        "functional_blocks",
        "previous_summaries",
        "previous_blocks",
    ],
    "A7__implement": [
        "current_plan",
        "initial_plan",
        "target_block",
        "current_solution",
    ],
    "A_verify": ["refined_code"],
    "A_sast": ["refined_code"],
    "eval_refinement": [
        "candidate_solution",
        "best_score",
        "debug_retries",
        "inner_step",
    ],
    "A8__plan": ["target_block", "inner_step", "execution_score"],
    "A11__debug_refine": ["refined_code", "debug_retries", "execution_error"],
    "Algorithm2": ["best_score", "status"],
}

_NODE_OUTPUT_FIELDS = {
    "A1__retrieve": ["retrieved_models"],
    "A2__generate": ["code"],
    "A3__merge": ["merged_code"],
    "A13__check_usage": ["code", "status"],
    "A13__fix_usage": ["code", "status"],
    "A12__check_leakage": ["code", "status"],
    "A12__fix_leakage": ["code", "status"],
    "eval_candidate": ["score", "status"],
    "A11__debug": ["code", "attempts", "status"],
    "A11__debug_merge": ["merged_code", "attempts", "status"],
    "A12__check_leakage_merge": ["merged_code", "status"],
    "A12__fix_leakage_merge": ["merged_code", "status"],
    "eval_merge": ["score", "status"],
    "generate_candidates": ["candidates_pool"],
    "candidate_flow": ["candidates_pool"],
    "Rank": ["leaderboard", "best_candidate"],
    "merge_candidates": ["candidates_pool"],
    "merge_flow": ["candidates_pool"],
    "SelectBest": ["best_candidate", "status"],
    "A4__generate_ablation": ["ablation_scripts", "functional_blocks", "status"],
    "ablation_variant_flow": ["ablation_results_list"],
    "eval_ablation_variant": ["execution_score", "execution_error", "status"],
    "A11__debug_ablation_variant": ["variant_code", "attempts", "status"],
    "A5__summarize_ablation": ["ablation_summaries"],
    "A6__extract_block": ["target_block", "initial_plan", "status"],
    "A7__implement": ["refined_code", "candidate_solution", "status"],
    "A_verify": ["status"],
    "A_sast": ["status"],
    "eval_refinement": [
        "execution_output",
        "execution_error",
        "execution_score",
        "status",
    ],
    "A8__plan": ["current_plan", "inner_step"],
    "A11__debug_refine": [
        "refined_code",
        "candidate_solution",
        "debug_retries",
        "status",
    ],
    "Algorithm2": ["improved_score", "improved_solution", "outer_step", "status"],
}


def generate_run_dir(base_root: str | None = None) -> str:
    """Generate a new timestamped run directory path.

    Format: {base_root}/YYYYMMDDHHMMSS_{uuid8}
    Default base_root is the project runs/ directory.
    """
    root = base_root or _RUNS_ROOT
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return os.path.join(root, f"{timestamp}_{short_id}")


def get_run_dir() -> str:
    """Get the current run directory.

    Priority:
        1. _current_run_dir context var (set by algorithm_1.run())
        2. sys.argv[1] (CLI argument, only if it looks like a run directory)
        3. Auto-generate a timestamped directory under runs/
    """
    ctx_dir = _current_run_dir.get()
    if ctx_dir is not None:
        os.makedirs(ctx_dir, exist_ok=True)
        return ctx_dir
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return sys.argv[1]
    run_dir = generate_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_checkpointer(run_dir: str):
    """Create a checkpointer for the given run directory.

    Returns SqliteSaver (file-based, cross-process persistence) when
    langgraph-checkpoint-sqlite is installed AND runtime-compatible.
    Falls back to InMemorySaver (in-process only) otherwise.

    Runtime compatibility: SqliteSaver 2.x may be incompatible with
    langgraph-checkpoint 4.x (serde API mismatch). This function detects
    the incompatibility at import time via _SQLITE_RUNTIME_COMPATIBLE and
    falls back to InMemorySaver.
    """
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if _SQLITE_AVAILABLE and _SQLITE_RUNTIME_COMPATIBLE:
        try:
            checkpointer = get_sqlite_checkpointer(checkpoint_dir)
            log_node_event(
                "checkpointer",
                "selected",
                {
                    "type": "SqliteSaver",
                    "sqlite_available": True,
                    "runtime_compatible": True,
                    "checkpoint_dir": checkpoint_dir,
                },
            )
            return checkpointer
        except Exception:
            log_node_event(
                "checkpointer",
                "fallback",
                {
                    "type": "InMemorySaver",
                    "sqlite_available": True,
                    "runtime_compatible": True,
                    "reason": "get_sqlite_checkpointer() failed",
                },
            )
    else:
        log_node_event(
            "checkpointer",
            "fallback",
            {
                "type": "InMemorySaver",
                "sqlite_available": _SQLITE_AVAILABLE,
                "runtime_compatible": _SQLITE_RUNTIME_COMPATIBLE,
                "reason": (
                    "SqliteSaver not runtime-compatible"
                    if _SQLITE_AVAILABLE
                    else "langgraph-checkpoint-sqlite not installed"
                ),
            },
        )
    return InMemorySaver()


def get_sqlite_checkpointer(checkpoint_dir: str):
    """Create a file-based SqliteSaver checkpointer.

    Uses sqlite3.connect() directly since from_conn_string() may fail
    on platforms where sqlite-vec wheels are unavailable (e.g., Alpine/musl).

    Args:
        checkpoint_dir: Directory to store the SQLite database file.

    Returns:
        SqliteSaver instance backed by a file-based SQLite database.

    Raises:
        ImportError: If langgraph-checkpoint-sqlite is not installed.
    """
    import sqlite3

    if not _SQLITE_AVAILABLE:
        raise ImportError(
            "langgraph-checkpoint-sqlite is not installed. "
            "Install it or use get_checkpointer() for InMemorySaver fallback."
        )

    db_path = os.path.join(checkpoint_dir, "graph_state.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def get_thread_id(run_dir: str) -> str:
    """Derive a thread ID from the run directory path."""
    return os.path.basename(run_dir.rstrip("/"))


def _trace_serialize(obj, budget=10000):
    """Serialize a dict for Langfuse tracing with full content preservation."""
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k in _TRACE_EXCLUDE_KEYS:
            if isinstance(v, list):
                out[k] = f"[{len(v)} events]"
            else:
                out[k] = f"[{v}]"
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, (int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = [_trace_serialize_item(item) for item in v]
        elif isinstance(v, dict):
            out[k] = _trace_serialize(v, budget=0)
        else:
            out[k] = str(v)[:1000]
    return out


def _trace_serialize_item(item, max_str=1000):
    """Serialize a single list item for Langfuse tracing."""
    if isinstance(item, dict):
        return _trace_serialize(item, budget=0)
    elif isinstance(item, str):
        return item if len(item) <= max_str else item[:max_str] + "..."
    elif isinstance(item, (int, float, bool)):
        return item
    else:
        return str(item)[:max_str]


def _extract_node_input(stage_name, state):
    """Extract relevant input fields for a node as a meaningful delta."""
    if not isinstance(state, dict):
        return str(state)[:1000]
    fields = _NODE_INPUT_FIELDS.get(stage_name, list(state.keys()))
    extracted = {k: state[k] for k in fields if k in state}
    return _trace_serialize(extracted)


def _extract_node_output(stage_name, result):
    """Extract relevant output fields for a node as a meaningful delta."""
    if not isinstance(result, dict):
        return str(result)[:1000]
    fields = _NODE_OUTPUT_FIELDS.get(stage_name, list(result.keys()))
    extracted = {k: result[k] for k in fields if k in result}
    return _trace_serialize(extracted)


def traceable(stage_name: str):
    """Decorator that adds Langfuse tracing to a node function.

    Creates a nested observation (span) under the current parent observation.
    Each node appears as a child span in the Langfuse flow hierarchy.

    Logs entry/exit with duration_ms as structured JSON via log_node_event,
    always written to runs/_dev/dev.log for troubleshooting.

    Flushing is controlled by LANGFUSE_FLUSH_PERIOD env var:
        - Not set or empty: No per-node flush (root-only at run end).
        - "0": Flush after every observation.
        - Positive float (e.g. "30"): Flush if this many seconds elapsed.

    Args:
        stage_name: Name of the stage/agent (e.g., 'A1__retrieve', 'A2__generate').
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            parent_obs = _current_obs.get()
            input_data = _extract_node_input(stage_name, state)
            obs = None
            obs_token = None

            log_node_event(
                stage_name,
                "traceable_enter",
                {"has_parent": parent_obs is not None},
            )

            try:
                if parent_obs is not None:
                    obs = parent_obs.start_observation(
                        name=stage_name,
                        input=input_data,
                    )
                else:
                    obs = langfuse.start_observation(
                        name=stage_name,
                        input=input_data,
                    )
            except Exception:
                log_node_event(
                    stage_name,
                    "traceable_error",
                    {"error": "failed to start observation"},
                )

            if obs is not None:
                obs_token = _current_obs.set(obs)
                log_node_event(
                    stage_name,
                    "traceable_obs_created",
                    {"obs_id": obs.id[:12] if hasattr(obs, "id") else "?"},
                )

            start = time.time()
            result = fn(state, *args, **kwargs)
            duration_ms = (time.time() - start) * 1000

            output_data = _extract_node_output(stage_name, result)

            if obs is not None:
                try:
                    obs.update(output=output_data)
                    obs.end()
                except Exception:
                    log_node_event(
                        stage_name,
                        "traceable_error",
                        {"error": "failed to end observation"},
                    )
                if _should_flush():
                    try:
                        langfuse.flush()
                    except Exception:
                        log_node_event(
                            stage_name,
                            "traceable_error",
                            {"error": "flush failed"},
                        )

            if obs_token is not None:
                _current_obs.reset(obs_token)

            log_node_event(stage_name, "output", output_data, duration_ms=duration_ms)

            log_node_event(
                stage_name,
                "traceable_exit",
                {"duration_ms": round(duration_ms, 2)},
            )

            if isinstance(state, dict):
                if "stage_history" not in state:
                    state["stage_history"] = []
                state["stage_history"].append(
                    {
                        "stage": stage_name,
                        "model": state.get("model")
                        if isinstance(state, dict)
                        else None,
                        "status": "complete",
                        "duration_ms": round(duration_ms, 2),
                        "timestamp": time.strftime("%Y-%m-%dT%H;%M:%S"),
                        "sub_events": [],
                    }
                )

            return result

        return wrapper

    return decorator


class SubgraphSpan:
    """Context manager for wrapping subgraph invocations with a parent span.

    Creates a span observation in Langfuse so all child nodes inside the
    subgraph appear grouped under it in the Langfuse UI.

    Nesting priority:
        1. If _current_obs is set (parent @traceable or outer SubgraphSpan):
           create as child via parent_obs.start_observation()
        2. Else if trace_id/parent_span_id provided explicitly:
           create with TraceContext for cross-context nesting
        3. Else: create a root observation under a new trace
    """

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        input_data: Optional[Dict] = None,
    ):
        self.name = name
        self.model = model
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.input_data = input_data
        self._output_data = None
        self.observation = None
        self.config: Optional[Dict] = None
        self._token = None

    def set_output(self, output_data: Dict):
        """Set the output data for this span, recorded on exit."""
        self._output_data = output_data

    def __enter__(self):
        """Enter the span context: create observation and set context var."""
        try:
            parent_obs = _current_obs.get()

            start_kwargs = dict(
                name=self.name,
                as_type="span",
                metadata={
                    "subgraph": self.name,
                    "model": self.model,
                    "type": "subgraph_parent",
                },
            )
            if self.input_data is not None:
                start_kwargs["input"] = self.input_data

            if parent_obs is not None:
                self.observation = parent_obs.start_observation(**start_kwargs)
            elif self.trace_id and self.parent_span_id:
                from langfuse.types import TraceContext

                start_kwargs["trace_context"] = TraceContext(
                    trace_id=self.trace_id,
                    parent_span_id=self.parent_span_id,
                )
                self.observation = langfuse.start_observation(**start_kwargs)
            else:
                self.observation = langfuse.start_observation(**start_kwargs)

            self.trace_id = self.observation.trace_id
            self.span_id = self.observation.id

            self.config = {
                "configurable": {
                    "thread_id": f"{self.name}_{self.model or 'merge'}_{self.trace_id[:8]}",
                    "run_id": self.trace_id,
                }
            }

            self._token = _current_obs.set(self.observation)
        except Exception:
            log_node_event(
                self.name,
                "subgraph_span_error",
                {"error": "failed to start observation"},
            )
            self.config = {"configurable": {"thread_id": "default"}}

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the span context: update output, end observation, reset context var."""
        try:
            if self.observation is not None:
                if self._output_data is not None:
                    self.observation.update(output=self._output_data)
                self.observation.end()
            if self._token is not None:
                _current_obs.reset(self._token)
        except Exception:
            log_node_event(
                self.name,
                "subgraph_span_error",
                {"error": "failed to end observation"},
            )
        return False


def log_node_event(
    node_name: str,
    event_type: str,
    data: Dict | None = None,
    duration_ms: float | None = None,
    run_id: str | None = None,
    phase: str | None = None,
):
    """Log a structured JSON event for a node.

    Always writes to the dev rotating log (runs/_dev/dev.log) via the
    mle_star root logger. When MLELogger is active, also writes to the
    per-run log file and stdout.

    Args:
        node_name: Name of the node (e.g., 'A1__retrieve', 'A2__generate').
        event_type: Type of event ('input', 'output', 'error', 'transition',
                    'metric', 'llm_call', 'code_replace', 'state_snapshot').
        data: Optional dict of event data.
        duration_ms: Optional duration in milliseconds.
        run_id: Optional run ID override (uses _current_run_id context var if None).
        phase: Optional phase name override (uses _current_phase context var if None).
    """
    effective_run_id = run_id or _current_run_id.get()
    effective_phase = phase or _current_phase.get()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": effective_run_id,
        "node": node_name,
        "event": event_type,
        "phase": effective_phase,
        "data": data or {},
    }
    if duration_ms is not None:
        entry["duration_ms"] = round(duration_ms, 2)

    # Always write to dev log via root mle_star logger
    _logger.info(json.dumps(entry, default=str, ensure_ascii=False))

    # Also write to per-run log if MLELogger is active
    mle = _current_mle_logger.get()
    if mle is not None:
        mle.info(
            node_name,
            event_type,
            data or {},
            duration_ms=duration_ms,
            run_id=run_id,
            phase=phase,
        )


METRIC_DIRECTIONS = {
    "accuracy": "maximize",
    "auc": "maximize",
    "roc_auc": "maximize",
    "f1": "maximize",
    "f1_score": "maximize",
    "f1_macro": "maximize",
    "f1_micro": "maximize",
    "f1_weighted": "maximize",
    "r2": "maximize",
    "ap": "maximize",
    "average_precision": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "sensitivity": "maximize",
    "specificity": "maximize",
    "ndcg": "maximize",
    "map": "maximize",
    "auroc": "maximize",
    "rmsle": "minimize",
    "root mean squared logarithmic error": "minimize",
    "mean_squared_logarithmic_error": "minimize",
    "mean absolute error": "minimize",
    "mean_absolute_error": "minimize",
    "mean squared error": "minimize",
    "rmse": "minimize",
    "mse": "minimize",
    "mae": "minimize",
    "log_loss": "minimize",
    "logloss": "minimize",
    "binary_crossentropy": "minimize",
    "categorical_crossentropy": "minimize",
    "hinge": "minimize",
    "kl_divergence": "minimize",
    "poisson": "minimize",
}


def infer_metric_direction(score_function_desc: str) -> str:
    """Infer whether higher or lower scores are better from the metric description.

    Checks for known metric keywords in the description. Keywords are matched
    longest-first to avoid short keywords matching substrings of longer ones
    (e.g., 'ap' matching inside 'rmsle' or 'map'). Falls back to 'maximize'.
    """
    desc_lower = score_function_desc.lower()
    for keyword, direction in sorted(
        METRIC_DIRECTIONS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if keyword in desc_lower:
            return direction
    return "maximize"


def normalize_score(raw_score: float, direction: str) -> float:
    """Normalize score so that higher is always better.

    For 'minimize' metrics, negate the score.
    """
    if direction == "minimize":
        return -raw_score
    return raw_score


def display_score(normalized_score: float, direction: str) -> float:
    """Convert normalized score back to original units for display.

    For 'minimize' metrics, negate back.
    """
    if direction == "minimize":
        return -normalized_score
    return normalized_score


def failure_score(metric_direction: str) -> float:
    """Return a score representing complete failure for the given direction.

    For 'minimize' (lower is better), returns inf (worst possible).
    For 'maximize' (higher is better), returns 0.0 (worst possible).
    After normalize_score(), both produce the smallest value,
    ensuring failed candidates never beat real ones.
    """
    if metric_direction == "minimize":
        return float("inf")
    return 0.0


def format_direction(metric_direction: str) -> str:
    """Convert metric_direction to a human-readable description for prompts.

    Args:
        metric_direction: 'minimize' or 'maximize'

    Returns:
        'lower is better = minimize' or 'higher is better = maximize'
    """
    if metric_direction == "minimize":
        return "lower is better = minimize"
    return "higher is better = maximize"


def random_score(min_val: float = 0.05, max_val: float = 0.08) -> float:
    """Generate a random score within the given range."""
    import random as _random

    return round(_random.uniform(min_val, max_val), 4)


def random_pass(prob: float = 0.9) -> bool:
    """Determine if a random check passes based on probability."""
    import random as _random

    return _random.random() < prob


def simulate_delay(min_s: float = 0.0, max_s: float = 0.0):
    """No-op placeholder. Simulated delays disabled for development speed.

    Previously slept for a random duration between min_s and max_s seconds.
    Will be re-enabled when real LLM calls need timing simulation.
    """
    pass


def shutdown_langfuse():
    """Flush and shut down the Langfuse client."""
    log_node_event("langfuse", "shutdown_start", {})
    start = time.time()
    try:
        langfuse.shutdown()
    except Exception:
        log_node_event(
            "langfuse", "shutdown_error", {"error": "langfuse.shutdown() failed"}
        )
    elapsed_ms = round((time.time() - start) * 1000, 2)
    log_node_event("langfuse", "shutdown_end", {"duration_ms": elapsed_ms})


# ── LLM Infrastructure ────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    """Configuration for LLM provider selection and parameters.

    Parameters are read from environment variables with sensible defaults.
    The primary provider is Ollama Cloud via langchain-ollama (ChatOllama).
    OpenAI-compatible providers (openrouter, openai) use httpx directly.
    """

    provider: str = "ollama"
    model: str = "glm-5.1:cloud"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 300

    def __post_init__(self):
        if self.base_url is None:
            if self.provider == "ollama":
                self.base_url = os.environ.get(
                    "OLLAMA_BASE_URL", "https://api.ollama.com"
                )
            elif self.provider == "openrouter":
                self.base_url = os.environ.get(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                )
            elif self.provider == "openai":
                self.base_url = os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                )
        if self.api_key is None:
            if self.provider == "ollama":
                self.api_key = os.environ.get("OLLAMA_API_KEY", "")
            elif self.provider == "openrouter":
                self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
            elif self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY", "")


def _default_llm_config() -> LLMConfig:
    """Create an LLMConfig from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        if provider == "ollama":
            model = os.environ.get("OLLAMA_MODEL", "glm-5.1:cloud")
        elif provider == "openrouter":
            model = os.environ.get(
                "OPENROUTER_MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free"
            )
        elif provider == "openai":
            model = "gpt-4o"
        else:
            model = "glm-5.1:cloud"
    return LLMConfig(provider=provider, model=model)


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    response_format: str = "text",
    config: Optional["LLMConfig"] = None,
) -> str:
    """Call an LLM with the given prompt.

    Provider routing:
        - "ollama": Ollama Cloud via langchain-ollama (ChatOllama)
        - "openrouter": OpenRouter API (OpenAI-compatible via httpx)
        - "openai": OpenAI API (OpenAI-compatible via httpx)

    On failure with the primary provider, falls back to OpenRouter if
    OPENROUTER_API_KEY is available.

    Args:
        prompt: The user prompt.
        system_prompt: Optional system prompt.
        response_format: Expected response format for logging ("text", "json", "code").
        config: LLM configuration. Falls back to _default_llm_config().

    Returns:
        The LLM response as a string.
    """
    if config is None:
        config = _default_llm_config()

    _increment_and_check_llm_limit()

    start_time = time.time()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        if config.provider == "ollama":
            response_text = _call_chat_ollama(messages, config)
        elif config.provider in ("openrouter", "openai"):
            response_text = _call_openai_compatible(messages, config)
        else:
            raise ValueError(f"Unknown LLM provider: {config.provider}")
    except Exception as primary_error:
        fallback_provider = os.environ.get("MLE_LLM_FALLBACK_PROVIDER", "").lower()
        fallback_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not fallback_provider and fallback_key:
            fallback_provider = "openrouter"

        if fallback_provider and fallback_provider != config.provider:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            log_node_event(
                "call_llm",
                "primary_failed_trying_fallback",
                {
                    "provider": config.provider,
                    "model": config.model,
                    "error": str(primary_error)[:300],
                    "fallback_provider": fallback_provider,
                    "duration_ms": duration_ms,
                },
            )
            try:
                fallback_config = _build_fallback_config(fallback_provider)
                if fallback_provider == "ollama":
                    response_text = _call_chat_ollama(messages, fallback_config)
                else:
                    response_text = _call_openai_compatible(messages, fallback_config)
            except Exception as fallback_error:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                log_node_event(
                    "call_llm",
                    "fallback_failed",
                    {
                        "primary_provider": config.provider,
                        "fallback_provider": fallback_provider,
                        "primary_error": str(primary_error)[:200],
                        "fallback_error": str(fallback_error)[:200],
                        "duration_ms": duration_ms,
                    },
                )
                raise primary_error
        else:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            log_node_event(
                "call_llm",
                "error",
                {
                    "provider": config.provider,
                    "model": config.model,
                    "error": str(primary_error)[:500],
                    "duration_ms": duration_ms,
                },
            )
            raise primary_error

    duration_ms = round((time.time() - start_time) * 1000, 2)
    log_node_event(
        "call_llm",
        "llm_call",
        {
            "provider": config.provider,
            "model": config.model,
            "prompt_length": len(prompt),
            "response_length": len(response_text),
            "duration_ms": duration_ms,
            "response_format": response_format,
        },
    )

    return response_text


def _build_fallback_config(provider: str) -> "LLMConfig":
    """Build an LLMConfig for a fallback provider from environment variables."""
    if provider == "openrouter":
        return LLMConfig(
            provider="openrouter",
            model=os.environ.get(
                "OPENROUTER_MODEL_NAME",
                "nvidia/nemotron-3-super-120b-a12b:free",
            ),
        )
    elif provider == "ollama":
        return LLMConfig(
            provider="ollama",
            model=os.environ.get("OLLAMA_MODEL", "glm-5.1:cloud"),
        )
    elif provider == "openai":
        return LLMConfig(
            provider="openai",
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )
    return LLMConfig(provider=provider)


def _call_chat_ollama(messages: list, config: "LLMConfig") -> str:
    """Call Ollama Cloud via langchain-ollama (ChatOllama).

    Uses the ChatOllama integration which talks to Ollama's native /api/chat
    endpoint. Supports cloud-hosted models with API key authentication.

    For reasoning models (e.g., glm-5.1:cloud), when the main content is
    empty but reasoning_content exists in additional_kwargs, extracts
    the reasoning as the response (which may contain the actual answer).
    Otherwise retries once with a follow-up message to extract the answer.
    """
    from langchain_ollama import ChatOllama

    client_headers = {}
    if config.api_key:
        client_headers["Authorization"] = f"Bearer {config.api_key}"

    llm = ChatOllama(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        num_predict=config.max_tokens,
        client_kwargs={"headers": client_headers} if client_headers else {},
    )

    lc_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lc_messages.append((role, content))

    response = llm.invoke(lc_messages)
    content = response.content

    if not content:
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            content = reasoning

    if not content:
        follow_up = lc_messages + [
            (
                "user",
                "Your previous response was empty. Please provide your answer directly without reasoning steps.",
            )
        ]
        retry_response = llm.invoke(follow_up)
        content = retry_response.content
        if not content:
            reasoning = retry_response.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                content = reasoning

    if not content:
        raise ValueError(f"Empty content from ChatOllama after retry: {response}")

    return content


def _call_openai_compatible(messages: list, config: "LLMConfig") -> str:
    """Call an OpenAI-compatible API (OpenRouter, OpenAI).

    Uses httpx for synchronous HTTP requests. The API is expected to follow
    the OpenAI chat completions format. Ollama provider is handled separately
    by _call_chat_ollama which uses langchain-ollama (ChatOllama).
    """
    import httpx

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    with httpx.Client(timeout=config.timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    if "choices" not in data or not data["choices"]:
        raise ValueError(f"No choices in LLM response: {data}")

    content = data["choices"][0].get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Empty content in LLM response: {data}")

    return content


# ── Parsing Utilities ─────────────────────────────────────────────────────


def parse_score(output: str) -> Optional[float]:
    """Extract 'Final Validation Performance: {score}' from execution output.

    Tries multiple patterns in order:
        1. Final Validation Performance: {score}
        2. Final Score: {score}
        3. Validation Score: {score}
        4. Score: {score}
        5. Fallback: last floating-point number in output

    Args:
        output: The stdout/stderr from subprocess execution.

    Returns:
        Extracted score as float, or None if no score found.
    """
    if not output:
        return None

    patterns = [
        r"Final\s+Validation\s+Performance:\s*([\d.]+)",
        r"Final\s+Score:\s*([\d.]+)",
        r"Validation\s+Score:\s*([\d.]+)",
        r"Score:\s*([\d.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    numbers = re.findall(r"[\d]+\.[\d]+", output)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


def parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response that may contain markdown fences.

    Tries in order:
        1. Direct JSON parse
        2. Extract from ```json ... ``` block (with flexible whitespace)
        3. Extract from ``` ... ``` block (any language)
        4. Handle unclosed ```json ... (no closing ```)
        5. Find first balanced { ... } using brace-depth tracking

    Args:
        raw: The raw LLM response text.

    Returns:
        Parsed JSON as a dict.

    Raises:
        ValueError: If no valid JSON could be extracted.
    """
    import json as _json

    if not raw or not raw.strip():
        raise ValueError("Empty LLM response")

    text = raw.strip()

    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return _json.loads(match.group(1).strip())
        except _json.JSONDecodeError:
            pass

    unclosed = re.search(r"```(?:json)?\s*\n(.*)", text, re.DOTALL)
    if unclosed:
        try:
            return _json.loads(unclosed.group(1).strip())
        except _json.JSONDecodeError:
            pass

    brace_depth = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                try:
                    return _json.loads(text[start_idx : i + 1])
                except _json.JSONDecodeError:
                    start_idx = None

    raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}...")


def parse_code_block(raw: str) -> str:
    """Extract Python code from LLM response that may contain markdown fences.

    Tries in order:
        1. Extract from ```python ... ``` block
        2. Extract from ``` ... ``` block (any language or no language)
        3. Extract from unclosed ```python ... (LLM truncation)
        4. If the raw text looks like code (has import/def/class/if __name__),
           return it as-is

    Args:
        raw: The raw LLM response text.

    Returns:
        Extracted code as a string.

    Raises:
        ValueError: If no code block could be extracted.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response")

    text = raw.strip()

    match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        lines = content.split("\n")
        if (
            lines
            and not lines[0].strip().startswith("#")
            and not lines[0].strip().startswith("import")
        ):
            pass
        return content

    # Handle unclosed fences: ```python\n... (no closing ```)
    # This occurs when LLM output is truncated or omits the closing fence.
    match = re.search(r"```(?:python|py)\s*\n(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    code_indicators = [
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__",
        "print(",
        "np.",
        "pd.",
        "sklearn.",
    ]
    if any(indicator in text for indicator in code_indicators):
        return text

    raise ValueError(f"Could not parse code block from LLM response: {raw[:200]}...")


def parse_code_blocks(solution: str) -> list[dict]:
    """Parse a Python solution into functional code blocks using AST.

    Returns:
        list of {name: str, code: str, start_line: int, end_line: int, type: str}
        for each top-level function, class, or assignment.
    """
    import ast as _ast

    blocks = []
    try:
        tree = _ast.parse(solution)
    except SyntaxError:
        lines = solution.splitlines()
        if lines:
            blocks.append(
                {
                    "name": "full_module",
                    "code": solution,
                    "start_line": 1,
                    "end_line": len(lines),
                    "type": "Module",
                }
            )
        return blocks

    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            code = _ast.get_source_segment(solution, node)
            blocks.append(
                {
                    "name": node.name,
                    "code": code or "",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno or node.lineno + 1,
                    "type": type(node).__name__,
                }
            )
        elif isinstance(node, _ast.ClassDef):
            code = _ast.get_source_segment(solution, node)
            blocks.append(
                {
                    "name": node.name,
                    "code": code or "",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno or node.lineno + 1,
                    "type": "ClassDef",
                }
            )
        elif isinstance(node, _ast.Assign):
            names = []
            for target in node.targets:
                if isinstance(target, _ast.Name):
                    names.append(target.id)
            code = _ast.get_source_segment(solution, node)
            blocks.append(
                {
                    "name": names[0] if names else "assign",
                    "code": code or "",
                    "start_line": node.lineno,
                    "end_line": node.end_lineno or node.lineno + 1,
                    "type": "Assignment",
                }
            )

    return blocks


def replace_code_block(solution: str, old_block: str, new_block: str) -> str:
    """Replace a code block in the solution using AST, exact, or fuzzy matching.

    Strategy:
        1. Try exact string match (cheapest)
        2. Try AST-based matching (handles LLM whitespace/formatting variations)
        3. Fall back to fuzzy replacement using difflib

    Args:
        solution: The full solution code.
        old_block: The code block to be replaced.
        new_block: The replacement code.

    Returns:
        The solution with old_block replaced by new_block.
    """
    import ast as _ast
    import difflib as _difflib

    if not old_block or not old_block.strip():
        return solution

    if old_block in solution:
        return solution.replace(old_block, new_block, 1)

    try:
        old_ast = _ast.parse(old_block)
        solution_lines = solution.splitlines(keepends=True)
        solution_tree = _ast.parse(solution)

        match_lengths = []
        for node in _ast.walk(solution_tree):
            if _ast.dump(node) == _ast.dump(old_ast):
                start = node.lineno - 1
                end = node.end_lineno
                match_lengths.append((start, end, end - start))

        match_lengths.sort(key=lambda x: x[2], reverse=True)

        for start, end, length in match_lengths:
            old_lines = solution_lines[start:end]
            new_lines = new_block.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            result_lines = solution_lines[:start] + new_lines + solution_lines[end:]
            return "".join(result_lines)

    except SyntaxError:
        pass

    return fuzzy_replace(solution, old_block, new_block)


def fuzzy_replace(solution: str, old_block: str, new_block: str) -> str:
    """Fuzzy string replacement using difflib.

    Finds the closest match to old_block among the double-newline-separated
    blocks in solution and replaces it. Multiple fallback strategies are used
    to tolerate code drift across refinement cycles.

    Args:
        solution: The full solution code.
        old_block: The code block to be replaced.
        new_block: The replacement code.

    Returns:
        The solution with the closest match replaced, or the original
        solution unchanged if no match could be found.
    """
    import difflib as _difflib

    if not old_block or not old_block.strip():
        return solution

    blocks = solution.split("\n\n")
    matches = _difflib.get_close_matches(old_block.strip(), blocks, n=1, cutoff=0.6)

    if matches:
        matched = matches[0]
        if old_block.strip() in matched:
            return solution.replace(matched, new_block, 1)
        matched_lines = matched.strip().splitlines()
        old_lines = old_block.strip().splitlines()
        if len(matched_lines) == len(old_lines):
            sm = _difflib.SequenceMatcher(None, matched_lines, old_lines)
            if sm.ratio() >= 0.8:
                return solution.replace(matched, new_block, 1)
        sm = _difflib.SequenceMatcher(
            None, matched.splitlines(), old_block.strip().splitlines()
        )
        if sm.ratio() >= 0.75:
            return solution.replace(matched, new_block, 1)
        s = _difflib.SequenceMatcher(
            None,
            matched.splitlines(),
            old_lines if "old_lines" in dir() else old_block.strip().splitlines(),
        )
        if s.ratio() >= 0.75:
            return solution.replace(matched, new_block, 1)

    old_lines = old_block.strip().splitlines()
    lines = solution.splitlines()

    if len(old_lines) <= 2 or len(lines) <= len(old_lines):
        log_node_event(
            "fuzzy_replace",
            "no_match",
            {"reason": "block_too_short", "old_preview": old_block[:80]},
        )
        return solution

    s = _difflib.SequenceMatcher(None, lines, old_lines)
    match = s.find_longest_match(0, len(lines), 0, len(old_lines))
    if match.size >= len(old_lines) * 0.6:
        replacement_lines = new_block.splitlines()
        result_lines = (
            lines[: match.a] + replacement_lines + lines[match.a + match.size :]
        )
        return "\n".join(result_lines)

    first_old_line = old_lines[0].strip() if old_lines else ""
    if first_old_line:
        for idx, line in enumerate(lines):
            if line.strip() == first_old_line:
                best_end = idx
                best_ratio = 0.0
                for end in range(idx + 1, min(idx + len(old_lines) + 5, len(lines))):
                    ratio = _difflib.SequenceMatcher(
                        None,
                        old_lines,
                        lines[idx:end],
                    ).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_end = end
                if best_ratio >= 0.6 and best_end > idx:
                    replacement_lines = new_block.splitlines()
                    result_lines = lines[:idx] + replacement_lines + lines[best_end:]
                    log_node_event(
                        "fuzzy_replace",
                        "first_line_match",
                        {
                            "old_preview": old_block[:80],
                            "insert_at": idx,
                            "replace_end": best_end,
                            "ratio": round(best_ratio, 3),
                        },
                    )
                    return "\n".join(result_lines)

    log_node_event(
        "fuzzy_replace",
        "no_match",
        {"reason": "all_strategies_failed", "old_preview": old_block[:80]},
    )
    return solution
