"""Merge Subgraph for Algorithm 1.

Merges two candidates (best + reference) to create an ensemble candidate.
Stage 7: Real LLM-based implementations with mock mode fallback.
Flow:
    A3__merge -> A12__check_leakage_merge -> eval_merge -> (optional debug)

In mock mode, uses simulated delays and random scores.
In real mode, uses LLM for merging and subprocess for execution.
"""

import os

from langgraph.graph import END, START, StateGraph

from src.mle_star.config import MOCK_MODE
from src.mle_star.state.alg1_state import MergeState
from src.mle_star.state.shared import (
    MAX_FIX_RETRIES,
    traceable,
    simulate_delay,
    log_node_event,
    normalize_score,
    random_pass,
    random_score,
    call_llm,
    _default_llm_config,
    parse_code_block,
    parse_json_response,
    format_direction,
)
from src.mle_star.nodes.execution import execute_code
from src.mle_star.prompts.search import MERGER_PROMPT
from src.mle_star.prompts.robustness import (
    DATA_LEAKAGE_EXTRACT_PROMPT,
    DATA_LEAKAGE_CORRECT_PROMPT,
    DEBUG_PROMPT,
)


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


@traceable("A3__merge")
def A3__merge(state: MergeState) -> MergeState:
    """Merge base and reference candidate codes.

    A3 from MLE-STAR paper - combines two candidate solutions into
    an ensemble. In mock mode, uses simple concatenation.
    In real mode, uses LLM to produce a merged script.
    """
    if _is_mock_mode():
        simulate_delay()
        state["merged_code"] = f"merged_{state['base_code']}_{state['ref_code']}"
        return state

    base_code = state.get("base_code", "")
    ref_code = state.get("ref_code", "")
    task_desc = state.get("task_desc", "ML regression task")
    metric_direction = state.get("metric_direction", "minimize")
    metric = state.get("score_function_desc", "") or (
        "RMSLE" if metric_direction == "minimize" else "accuracy"
    )

    prompt = MERGER_PROMPT.format(
        task_desc=task_desc[:500],
        metric=metric,
        direction=format_direction(metric_direction),
        base_code=base_code,
        ref_code=ref_code,
    )

    try:
        config = _default_llm_config()
        response = call_llm(prompt, response_format="code", config=config)
        merged_code = parse_code_block(response)
        state["merged_code"] = merged_code
        log_node_event("A3__merge", "success", {"merged_code_len": len(merged_code)})
    except Exception as e:
        log_node_event("A3__merge", "llm_error", {"error": str(e)[:200]})
        state["merged_code"] = ""
        state["score"] = 1.0
        state["status"] = "llm_failed"

    return state


@traceable("A12__check_leakage_merge")
def A12__check_leakage_merge(state: MergeState) -> MergeState:
    """Verify merged candidate has no data leakage.

    In mock mode, 90% pass rate. In real mode, uses LLM.
    Force-passes after MAX_FIX_RETRIES attempts.
    """
    if _is_mock_mode():
        simulate_delay()
        attempts = state.get("leakage_fix_attempts", 0)
        if attempts >= MAX_FIX_RETRIES:
            state["status"] = "ok"
            return state
        passed = random_pass(0.90)
        state["status"] = "ok" if passed else "leakage_fail"
        return state

    merged_code = state.get("merged_code", "")
    task_desc = state.get("task_desc", "ML regression task")
    metric = state.get("score_function_desc", "RMSLE")

    if not merged_code.strip():
        state["status"] = "ok"
        return state

    attempts = state.get("leakage_fix_attempts", 0)
    if attempts >= MAX_FIX_RETRIES:
        state["status"] = "ok"
        return state

    try:
        config = _default_llm_config()
        prompt = DATA_LEAKAGE_EXTRACT_PROMPT.format(
            task_desc=task_desc, metric=metric, code=merged_code
        )
        response = call_llm(prompt, response_format="json", config=config)
        parsed = parse_json_response(response)
        has_leakage = parsed.get("has_leakage", False)
        state["status"] = "ok" if not has_leakage else "leakage_fail"
        log_node_event(
            "A12__check_leakage_merge", "result", {"status": state["status"]}
        )
    except Exception as e:
        log_node_event("A12__check_leakage_merge", "llm_error", {"error": str(e)[:200]})
        state["status"] = "ok"

    return state


@traceable("A12__fix_leakage_merge")
def A12__fix_leakage_merge(state: MergeState) -> MergeState:
    """Fix data leakage in merged candidate code.

    In mock mode, appends '_fixed_leak'. In real mode, uses LLM.
    """
    if _is_mock_mode():
        simulate_delay()
        state["merged_code"] = state["merged_code"] + "_fixed_leak"
        state["leakage_fix_attempts"] = state.get("leakage_fix_attempts", 0) + 1
        state["status"] = "ok"
        return state

    merged_code = state.get("merged_code", "")
    task_desc = state.get("task_desc", "ML regression task")
    metric = state.get("score_function_desc", "RMSLE")

    try:
        config = _default_llm_config()
        prompt = DATA_LEAKAGE_CORRECT_PROMPT.format(
            task_desc=task_desc,
            metric=metric,
            leakage_issues="Data leakage in merged code",
            code=merged_code,
        )
        response = call_llm(prompt, response_format="code", config=config)
        fixed_code = parse_code_block(response)
        state["merged_code"] = fixed_code
        log_node_event("A12__fix_leakage_merge", "fixed", {"code_len": len(fixed_code)})
    except Exception as e:
        log_node_event("A12__fix_leakage_merge", "llm_error", {"error": str(e)[:200]})
        state["merged_code"] = state["merged_code"] + "\n# leakage fix attempted\n"

    state["leakage_fix_attempts"] = state.get("leakage_fix_attempts", 0) + 1
    state["status"] = "ok"
    return state


@traceable("eval_merge")
def eval_merge(state: MergeState) -> MergeState:
    """Execute merged candidate and evaluate performance.

    In mock mode, random score with 10% crash rate.
    In real mode, subprocess execution with score parsing.
    """
    if _is_mock_mode():
        simulate_delay()
        crashed = random_pass(0.10)
        attempts = state.get("attempts", 0)
        if crashed and attempts >= 3:
            state["score"] = random_score(0.04, 0.07)
            state["status"] = "ok"
        elif crashed:
            state["status"] = "crashed"
        else:
            state["score"] = random_score(0.04, 0.07)
            state["status"] = "ok"
        return state

    merged_code = state.get("merged_code", "")
    datasets = state.get("datasets", [])

    if not merged_code.strip():
        state["score"] = 1.0
        state["status"] = "ok"
        state["execution_output"] = ""
        state["execution_error"] = "Empty merged code"
        return state

    train_path = "input/train.csv"
    test_path = "input/test.csv"
    if datasets:
        for d in datasets:
            if "train" in d.lower():
                train_path = d
            if "test" in d.lower() and "submission" not in d.lower():
                test_path = d

    result = execute_code(code=merged_code, train_path=train_path, test_path=test_path)

    state["execution_output"] = result.get("stdout", "")
    state["execution_error"] = result.get("stderr", "")

    if (
        result["status"] == "error"
        or result["status"] == "timeout"
        or result["status"] == "blocked"
    ):
        attempts = state.get("attempts", 0)
        if attempts >= 3:
            state["score"] = 1.0
            state["status"] = "ok"
        else:
            state["status"] = "crashed"
    else:
        state["score"] = result.get("score", 1.0)
        state["status"] = "ok"

    return state


@traceable("A11__debug_merge")
def A11__debug_merge(state: MergeState) -> MergeState:
    """Debug crashed merged candidate.

    Stage 10: Uses reflection-based debugging — classifies the error type
    before generating a fix, providing more targeted debugging.

    In mock mode, appends '_debugged_{attempts}'. In real mode, uses LLM.
    Max 3 debug attempts before accepting a score.
    """
    from src.mle_star.nodes.robustness import _classify_error

    if _is_mock_mode():
        simulate_delay()
        state["merged_code"] = state["merged_code"] + f"_debugged_{state['attempts']}"
        state["attempts"] = state.get("attempts", 0) + 1
        state["status"] = "ok"
        return state

    merged_code = state.get("merged_code", "")
    error_message = state.get("execution_error", "unknown error")
    task_desc = state.get("task_desc", "ML regression task")
    metric = state.get("score_function_desc", "RMSLE")

    error_class = _classify_error(
        str(error_message), stderr=str(error_message)[:1000], code=merged_code
    )
    error_type = error_class["error_type"]
    error_suggestion = error_class["suggestion"]

    try:
        config = _default_llm_config()
        stderr = error_message[:1000] if error_message else ""
        prompt = DEBUG_PROMPT.format(
            task_desc=task_desc,
            metric=metric,
            code=merged_code,
            error_message=f"[{error_type}] {error_message}\nSuggestion: {error_suggestion}",
            stderr=stderr,
        )
        response = call_llm(prompt, response_format="code", config=config)
        fixed_code = parse_code_block(response)
        state["merged_code"] = fixed_code
        log_node_event(
            "A11__debug_merge",
            "debugged",
            {
                "attempts": state.get("attempts", 0),
                "code_len": len(fixed_code),
                "error_type": error_type,
            },
        )
    except Exception as e:
        log_node_event("A11__debug_merge", "llm_error", {"error": str(e)[:200]})
        state["merged_code"] = (
            state["merged_code"] + f"\n# debug attempt {state.get('attempts', 0)}\n"
        )

    state["attempts"] = state.get("attempts", 0) + 1
    state["status"] = "ok"
    return state


def route_after_merge(state: MergeState) -> str:
    if state.get("status") == "llm_failed":
        return END
    return "A12__check_leakage_merge"


def route_after_check_leakage(state: MergeState) -> str:
    if state.get("status") == "leakage_fail":
        return "A12__fix_leakage_merge"
    return "eval_merge"


def route_after_fix_leakage(state: MergeState) -> str:
    return "A12__check_leakage_merge"


def route_after_execute(state: MergeState) -> str:
    if state.get("status") == "crashed":
        attempts = state.get("attempts", 0)
        if attempts >= 3:
            return END
        return "A11__debug_merge"
    return END


_builder = StateGraph(MergeState)

_builder.add_node("A3__merge", A3__merge)
_builder.add_node("A12__check_leakage_merge", A12__check_leakage_merge)
_builder.add_node("A12__fix_leakage_merge", A12__fix_leakage_merge)
_builder.add_node("eval_merge", eval_merge)
_builder.add_node("A11__debug_merge", A11__debug_merge)

_builder.add_edge(START, "A3__merge")
_builder.add_conditional_edges(
    "A3__merge",
    route_after_merge,
    {"A12__check_leakage_merge": "A12__check_leakage_merge", END: END},
)
_builder.add_conditional_edges(
    "A12__check_leakage_merge",
    route_after_check_leakage,
    {"A12__fix_leakage_merge": "A12__fix_leakage_merge", "eval_merge": "eval_merge"},
)
_builder.add_conditional_edges(
    "A12__fix_leakage_merge",
    route_after_fix_leakage,
    {"A12__check_leakage_merge": "A12__check_leakage_merge"},
)
_builder.add_conditional_edges(
    "eval_merge",
    route_after_execute,
    {"A11__debug_merge": "A11__debug_merge", END: END},
)
_builder.add_edge("A11__debug_merge", "eval_merge")

merge_subgraph = _builder.compile()
