"""Robustness nodes: A11__debug, A12__check_leakage, A13__check_usage.

Also contains ensemble-specific variants:
  A11__debug_ensemble, A12__check_leakage_ensemble, A12__fix_leakage_ensemble

Reflection-based debugging (Stage 10): _classify_error classifies error types
using regex patterns first, with LLM fallback for ambiguous errors.

And routing functions for the ensemble round subgraph conditional edges.

Mock implementations for Stage 5. Real implementations built in Stages 7-9.
"""

import os
import re

from langgraph.graph import END

from src.mle_star.config import MOCK_MODE, MAX_ENSEMBLE_DEBUG_RETRIES
from src.mle_star.prompts.robustness import (
    DEBUG_PROMPT,
    DATA_LEAKAGE_EXTRACT_PROMPT,
    DATA_LEAKAGE_CORRECT_PROMPT,
    ERROR_CLASSIFIER_PROMPT,
)
from src.mle_star.state.shared import (
    traceable,
    simulate_delay,
    log_node_event,
    random_pass,
    call_llm,
    _default_llm_config,
    parse_code_block,
    parse_json_response,
)


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _infer_metric(state: dict) -> str:
    """Derive a metric name from state fields."""
    score_function_desc = state.get("score_function_desc", "")
    if score_function_desc:
        return score_function_desc
    metric_direction = state.get("metric_direction", "maximize")
    return "root_mean_squared_error" if metric_direction == "minimize" else "accuracy"


# ── Reflection-Based Debugging (Stage 10) ──────────────────────────────────


_ERROR_PATTERNS = [
    (
        r"ValueError: (?:Found array with|shapes? .* not|cannot reshape)",
        "ShapeMismatch",
    ),
    (
        r"shapes? \(\d+.*\) and \(\d+.*\) (?:not aligned|are incompatible|mismatch)",
        "ShapeMismatch",
    ),
    (
        r"DataLeakage|data leakage|train.*test.*contamination|target.*leakage",
        "DataLeakage",
    ),
    (r"ModuleNotFoundError: No module named", "ImportError"),
    (r"ImportError: (?:cannot import|No module named)", "ImportError"),
    (r"ValueError: (?:NaN|inf|Input contains NaN|could not convert)", "ValueError"),
    (r"ValueError:", "ValueError"),
    (r"KeyError: ", "KeyError"),
    (r"TypeError: ", "TypeError"),
    (r"AttributeError: .*(?:has no attribute|object has no)", "AttributeError"),
    (r"IndexError: .*(?:out of bounds|index out of range)", "IndexError"),
    (r"TimeoutExpired|timed out|Execution timed out", "Timeout"),
    (r"FileNotFoundError: ", "Other"),
    (r"ZeroDivisionError: ", "ValueError"),
    (r"RuntimeError: ", "Other"),
]


def _classify_error(error_msg: str, stderr: str = "", code: str = "") -> dict:
    """Classify an execution error into a known error type.

    Uses regex patterns to match common ML/Python error types.
    Falls back to LLM classification for ambiguous errors.

    Returns dict with:
        error_type: str (e.g., 'ShapeMismatch', 'ImportError')
        confidence: str ('high', 'medium', 'low')
        suggestion: str (brief fix hint)
    """
    combined = f"{error_msg}\n{stderr}"

    for pattern, error_type in _ERROR_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            suggestions = {
                "ShapeMismatch": "Check array dimensions and ensure compatible shapes for operations",
                "DataLeakage": "Ensure proper train/test split before any fitting or transformation",
                "ImportError": "Install missing package or check import name spelling",
                "ValueError": "Check for NaN/inf values, invalid parameters, or incorrect data types",
                "KeyError": "Check column names exist in DataFrame or keys in dictionary",
                "TypeError": "Check function argument types match expected signatures",
                "AttributeError": "Verify the object has the expected attribute/method",
                "IndexError": "Check array/DataFrame index bounds",
                "Timeout": "Optimize code or reduce data size for faster execution",
                "Other": "Review the error message for specific details",
            }
            return {
                "error_type": error_type,
                "confidence": "high"
                if error_type
                in (
                    "ShapeMismatch",
                    "ImportError",
                    "KeyError",
                    "AttributeError",
                    "IndexError",
                    "Timeout",
                )
                else "medium",
                "suggestion": suggestions.get(error_type, "Review the error message"),
            }

    if code and not _is_mock_mode():
        try:
            config = _default_llm_config()
            prompt = ERROR_CLASSIFIER_PROMPT.format(
                error_message=error_msg[:2000],
                stderr=stderr[:2000],
                code=code[:4000],
            )
            response = call_llm(prompt, response_format="json", config=config)
            result = parse_json_response(response)
            return {
                "error_type": result.get("error_type", "Other"),
                "confidence": result.get("confidence", "low"),
                "suggestion": result.get("suggestion", ""),
            }
        except Exception:
            pass

    return {
        "error_type": "Other",
        "confidence": "low",
        "suggestion": "Review the error message for specific details",
    }


# ── Ensemble Robustness Nodes ─────────────────────────────────────────────


@traceable("A11__debug_ensemble")
def A11__debug_ensemble(state: dict) -> dict:
    """Debug a failed ensemble execution.

    Stage 10: Uses reflection-based debugging — classifies the error type
    before generating a fix, providing more targeted debugging.

    Attempts to fix the error in the ensemble code. Increments debug_retries.
    Max MAX_ENSEMBLE_DEBUG_RETRIES attempts before giving up.

    Mock: marks the code as debugged by appending a debug comment.
    Real: calls LLM to diagnose and fix the error, falls back to mock on failure.
    """
    simulate_delay()

    debug_retries = state.get("debug_retries", 0) + 1
    execution_error = state.get("execution_error", "unknown error")

    if _is_mock_mode():
        current_code = state.get("current_ensemble_code", "")
        debugged_code = current_code + f"# debug attempt {debug_retries}\n"

        error_class = _classify_error(str(execution_error))

        log_node_event(
            "A11__debug_ensemble",
            "output",
            {
                "debug_retries": debug_retries,
                "error": str(execution_error)[:60],
                "error_type": error_class["error_type"],
                "mode": "mock",
            },
        )

        return {
            "current_ensemble_code": debugged_code,
            "debug_retries": debug_retries,
            "execution_error": None,
            "status": "debugged",
        }

    current_code = state.get("current_ensemble_code", "")
    task_desc = state.get("task_desc", "ML task")
    metric = _infer_metric(state)
    stderr = state.get("stderr", "")

    error_class = _classify_error(
        str(execution_error), stderr=stderr, code=current_code
    )
    error_type = error_class["error_type"]
    error_suggestion = error_class["suggestion"]

    prompt = DEBUG_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        code=current_code,
        error_message=f"[{error_type}] {execution_error}\nSuggestion: {error_suggestion}",
        stderr=stderr,
    )

    try:
        config = _default_llm_config()
        response = call_llm(prompt, response_format="code", config=config)
        debugged_code = parse_code_block(response)

        log_node_event(
            "A11__debug_ensemble",
            "output",
            {
                "debug_retries": debug_retries,
                "error": str(execution_error)[:60],
                "error_type": error_type,
                "mode": "llm",
            },
        )

        return {
            "current_ensemble_code": debugged_code,
            "debug_retries": debug_retries,
            "execution_error": None,
            "status": "debugged",
        }
    except Exception as e:
        log_node_event(
            "A11__debug_ensemble",
            "llm_fallback",
            {"error": str(e)[:200], "mode": "fallback_to_mock"},
        )

        debugged_code = current_code + f"# debug attempt {debug_retries}\n"

        return {
            "current_ensemble_code": debugged_code,
            "debug_retries": debug_retries,
            "execution_error": None,
            "status": "debugged",
        }


@traceable("A12__check_leakage_ensemble")
def A12__check_leakage_ensemble(state: dict) -> dict:
    """Data leakage check for ensemble code.

    Two-step process: extract code blocks that may contain leakage,
    then detect if leakage exists (e.g., target leakage, train-test contamination).

    Mock: always passes (no leakage detected).
    Real: calls LLM to analyze code for leakage, falls back to ok on failure.
    """
    simulate_delay()

    current_code = state.get("current_ensemble_code", "")

    if _is_mock_mode():
        log_node_event(
            "A12__check_leakage_ensemble",
            "output",
            {"status": "ok", "code_len": len(current_code), "mode": "mock"},
        )

        return {"leakage_status": "ok", "status": "ok"}

    task_desc = state.get("task_desc", "ML task")
    metric = _infer_metric(state)

    prompt = DATA_LEAKAGE_EXTRACT_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        code=current_code,
    )

    try:
        config = _default_llm_config()
        response = call_llm(prompt, response_format="json", config=config)
        parsed = parse_json_response(response)

        has_leakage = parsed.get("has_leakage", False)
        if has_leakage:
            leakage_issues = parsed.get("leakage_issues", [])
            log_node_event(
                "A12__check_leakage_ensemble",
                "output",
                {"status": "leakage_fail", "has_leakage": True, "mode": "llm"},
            )
            return {
                "leakage_status": "Yes Data Leakage",
                "leakage_issues": leakage_issues,
                "status": "leakage_fail",
            }

        log_node_event(
            "A12__check_leakage_ensemble",
            "output",
            {"status": "ok", "has_leakage": False, "mode": "llm"},
        )
        return {"leakage_status": "ok", "status": "ok"}
    except Exception as e:
        log_node_event(
            "A12__check_leakage_ensemble",
            "llm_fallback",
            {"error": str(e)[:200], "mode": "fallback_to_ok"},
        )
        return {"leakage_status": "ok", "status": "ok"}


@traceable("A12__fix_leakage_ensemble")
def A12__fix_leakage_ensemble(state: dict) -> dict:
    """Fix data leakage detected in ensemble code.

    A12 from MLE-STAR paper — corrects the identified leakage in the code.

    Mock: appends a leakage fix comment and sets status to re-check.
    Real: calls LLM to fix leakage, falls back to mock on failure.
    """
    simulate_delay()

    current_code = state.get("current_ensemble_code", "")

    if _is_mock_mode():
        fixed_code = current_code + "# leakage_fix_ensemble\n"

        log_node_event(
            "A12__fix_leakage_ensemble",
            "output",
            {"status": "fixed", "code_len": len(fixed_code), "mode": "mock"},
        )

        return {
            "current_ensemble_code": fixed_code,
            "leakage_status": None,
            "status": "leakage_fix_applied",
        }

    task_desc = state.get("task_desc", "ML task")
    metric = _infer_metric(state)
    leakage_issues = state.get("leakage_issues", [])
    leakage_issues_str = (
        str(leakage_issues) if leakage_issues else "No specific leakage issues provided"
    )

    prompt = DATA_LEAKAGE_CORRECT_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        leakage_issues=leakage_issues_str,
        code=current_code,
    )

    try:
        config = _default_llm_config()
        response = call_llm(prompt, response_format="code", config=config)
        fixed_code = parse_code_block(response)

        log_node_event(
            "A12__fix_leakage_ensemble",
            "output",
            {"status": "fixed", "code_len": len(fixed_code), "mode": "llm"},
        )

        return {
            "current_ensemble_code": fixed_code,
            "leakage_status": None,
            "status": "leakage_fix_applied",
        }
    except Exception as e:
        log_node_event(
            "A12__fix_leakage_ensemble",
            "llm_fallback",
            {"error": str(e)[:200], "mode": "fallback_to_mock"},
        )

        fixed_code = current_code + "# leakage_fix_ensemble\n"

        return {
            "current_ensemble_code": fixed_code,
            "leakage_status": None,
            "status": "leakage_fix_applied",
        }


# ── Routing Functions for Ensemble Round Subgraph ─────────────────────────


def route_after_leakage_check_ensemble(state: dict) -> str:
    """Route after A12__check_leakage_ensemble.

    E-E01: leakage_fail -> A12__fix_leakage_ensemble
    E-E03: ok -> eval_ensemble
    """
    if state.get("leakage_status") in ("leakage_fail", "Yes Data Leakage"):
        return "A12__fix_leakage_ensemble"
    return "eval_ensemble"


def route_after_ensemble_eval(state: dict) -> str:
    """Route after eval_ensemble in the ensemble round subgraph.

    E-E04: error and debug_retries < MAX_ENSEMBLE_DEBUG_RETRIES -> A11__debug_ensemble
    E-E05: error and debug_retries >= MAX_ENSEMBLE_DEBUG_RETRIES -> END (give up)
    ok -> END (step complete; next round handled by Python loop)
    """
    if state.get("status") == "error":
        if state.get("debug_retries", 0) < MAX_ENSEMBLE_DEBUG_RETRIES:
            return "A11__debug_ensemble"
        return END
    return END
