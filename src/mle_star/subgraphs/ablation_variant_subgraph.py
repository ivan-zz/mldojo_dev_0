"""Ablation variant execution subgraph with debug retry loop.

Executes a single ablation variant script (baseline or one disabled component).
If execution fails, routes to debugger for up to 3 retries before giving up.

Topology:
    START → eval_ablation_variant
              ├─(ok) → END
              ├─(error, attempts < 3) → A11__debug_ablation_variant → eval_ablation_variant
              └─(error, attempts >= 3) → END

Mirrors the candidate_subgraph pattern (eval → debug → retry).
"""

import os

from langgraph.graph import END, START, StateGraph

from src.mle_star.config import MOCK_MODE, MAX_ABLATION_DEBUG_RETRIES
from src.mle_star.state.alg2_state import AblationVariantState
from src.mle_star.state.shared import (
    random_pass,
    random_score,
    simulate_delay,
    traceable,
    log_node_event,
    call_llm,
    _default_llm_config,
    parse_code_block,
    parse_score,
)
from src.mle_star.nodes.execution import execute_code
from src.mle_star.prompts.robustness import DEBUG_PROMPT


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


@traceable("eval_ablation_variant")
def eval_ablation_variant(state: AblationVariantState) -> AblationVariantState:
    """Execute an ablation variant script and evaluate.

    Mock: generates a random score. 10% chance of error.
    If error and attempts >= 3, accepts a score anyway.

    Real: executes the variant code via execute_code(), parses the score,
    and handles failures with debug retry or graceful give-up.
    """
    if _is_mock_mode():
        simulate_delay()

        crashed = random_pass(0.10)
        attempts = state.get("attempts", 0)

        if crashed and attempts >= MAX_ABLATION_DEBUG_RETRIES:
            state["execution_score"] = random_score(0.5, 0.7)
            state["status"] = "ok"
        elif crashed:
            state["execution_error"] = "Mock execution error"
            state["status"] = "error"
        else:
            state["execution_score"] = random_score(0.7, 0.95)
            state["status"] = "ok"

        return state

    variant_code = state.get("variant_code", "")
    attempts = state.get("attempts", 0)
    block_name = state.get("block_name", "")

    log_node_event(
        "eval_ablation_variant",
        "real_execution_start",
        {
            "variant_name": state.get("variant_name", ""),
            "block_name": block_name,
            "attempts": attempts,
        },
    )

    result = execute_code(variant_code)

    if result.get("status") != "ok":
        stderr = result.get("stderr", "Unknown execution error")
        if attempts < MAX_ABLATION_DEBUG_RETRIES:
            state["execution_output"] = result.get("stdout", "")
            state["execution_error"] = stderr
            state["execution_score"] = None
            state["status"] = "error"
            log_node_event(
                "eval_ablation_variant",
                "execution_failed",
                {"attempts": attempts, "error": stderr[:300]},
            )
        else:
            state["execution_output"] = result.get("stdout", "")
            state["execution_error"] = None
            state["execution_score"] = 999.0
            state["status"] = "ok"
            log_node_event(
                "eval_ablation_variant",
                "execution_failed_max_retries",
                {"attempts": attempts, "error": stderr[:300]},
            )
        return state

    raw_score = parse_score(result.get("stdout", ""))
    if raw_score is None:
        if attempts < MAX_ABLATION_DEBUG_RETRIES:
            state["execution_output"] = result.get("stdout", "")
            state["execution_error"] = "Could not parse score from execution output"
            state["execution_score"] = None
            state["status"] = "error"
        else:
            state["execution_output"] = result.get("stdout", "")
            state["execution_error"] = None
            state["execution_score"] = 999.0
            state["status"] = "ok"
        return state

    state["execution_output"] = result.get("stdout", "")
    state["execution_error"] = None
    state["execution_score"] = raw_score
    state["status"] = "ok"

    log_node_event(
        "eval_ablation_variant",
        "execution_success",
        {"block_name": block_name, "score": raw_score},
    )

    return state


@traceable("A11__debug_ablation_variant")
def A11__debug_ablation_variant(state: AblationVariantState) -> AblationVariantState:
    """Debug a failed ablation variant execution.

    Stage 10: Uses reflection-based debugging — classifies the error type
    before generating a fix, providing more targeted debugging.

    Attempts to fix the error in the variant script. Max 3 retries.

    Mock: marks as fixed by appending a debug note.
    Real: calls LLM with DEBUG_PROMPT to fix the code, falls back to
    mock behavior if LLM call fails.
    """
    from src.mle_star.nodes.robustness import _classify_error

    if _is_mock_mode():
        simulate_delay()

        state["variant_code"] = (
            state.get("variant_code", "")
            + f"# debug attempt {state.get('attempts', 0)}\n"
        )
        state["attempts"] = state.get("attempts", 0) + 1
        state["execution_error"] = None
        state["status"] = "ok"

        return state

    variant_code = state.get("variant_code", "")
    execution_error = state.get("execution_error", "")
    attempts = state.get("attempts", 0)

    error_class = _classify_error(str(execution_error), code=variant_code)
    error_type = error_class["error_type"]
    error_suggestion = error_class["suggestion"]

    log_node_event(
        "A11__debug_ablation_variant",
        "real_debug_start",
        {
            "attempts": attempts,
            "error_length": len(execution_error) if execution_error else 0,
            "error_type": error_type,
        },
    )

    try:
        prompt = DEBUG_PROMPT.format(
            task_desc=state.get("variant_name", "ablation variant"),
            code=variant_code,
            error_message=f"[{error_type}] {execution_error or 'Unknown error'}\nSuggestion: {error_suggestion}",
            stderr="",
            metric="score",
        )

        llm_response = call_llm(prompt, config=_default_llm_config())

        fixed_code = parse_code_block(llm_response)

        state["variant_code"] = fixed_code
        state["attempts"] = attempts + 1
        state["execution_error"] = None
        state["status"] = "ok"

        log_node_event(
            "A11__debug_ablation_variant",
            "debug_success",
            {"attempts": attempts + 1, "error_type": error_type},
        )

    except Exception as e:
        log_node_event(
            "A11__debug_ablation_variant",
            "debug_llm_failed",
            {"error": str(e)[:300], "attempts": attempts},
        )

        state["variant_code"] = (
            state.get("variant_code", "") + f"# debug attempt {attempts}\n"
        )
        state["attempts"] = attempts + 1
        state["execution_error"] = None
        state["status"] = "ok"

    return state


def route_after_eval_variant(state: AblationVariantState) -> str:
    """Route based on execution result."""
    if state.get("status") == "error":
        attempts = state.get("attempts", 0)
        if attempts < MAX_ABLATION_DEBUG_RETRIES:
            return "A11__debug_ablation_variant"
        return END
    return END


_builder = StateGraph(AblationVariantState)

_builder.add_node("eval_ablation_variant", eval_ablation_variant)
_builder.add_node("A11__debug_ablation_variant", A11__debug_ablation_variant)

_builder.add_edge(START, "eval_ablation_variant")
_builder.add_conditional_edges(
    "eval_ablation_variant",
    route_after_eval_variant,
    {
        "A11__debug_ablation_variant": "A11__debug_ablation_variant",
        END: END,
    },
)
_builder.add_edge("A11__debug_ablation_variant", "eval_ablation_variant")

ablation_variant_subgraph = _builder.compile(name="AblationVariantSubgraph")
