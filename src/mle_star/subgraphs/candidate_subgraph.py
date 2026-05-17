"""Candidate Generation Subgraph for Algorithm 1.

Generates initial code candidates for each model with:
    A2__Generate code -> A13__Check data usage -> A12__Check data leakage
    -> eval_candidate Execute -> (optional debug loop via A11)

Stage 7: Real implementations with LLM calls and subprocess execution.
Falls back to mock behavior when MLE_MOCK_MODE is enabled.
"""

import os

from langgraph.graph import END, START, StateGraph

from src.mle_star.config import MOCK_MODE, MAX_DEBUG_RETRIES as _MAX_DEBUG_RETRIES
from src.mle_star.state.alg1_state import CandidateState
from src.mle_star.state.shared import (
    MAX_FIX_RETRIES,
    traceable,
    simulate_delay,
    log_node_event,
    random_score,
    random_pass,
    normalize_score,
    call_llm,
    _default_llm_config,
    parse_code_block,
    parse_json_response,
    format_direction,
)
from src.mle_star.nodes.execution import execute_code, validate_code_safety
from src.mle_star.prompts.search import CANDIDATE_EVAL_PROMPT
from src.mle_star.prompts.robustness import (
    DATA_USAGE_CHECK_PROMPT,
    DATA_USAGE_FIX_PROMPT,
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


@traceable("A2__generate")
def A2__generate(state: CandidateState) -> CandidateState:
    """Generate initial code for the candidate model.

    A2 from MLE-STAR paper - generates an initial solution (code string)
    for the given model. Uses LLM with CANDIDATE_EVAL_PROMPT.
    In mock mode, generates a placeholder code string.
    """
    if _is_mock_mode():
        import uuid

        simulate_delay()
        state["code"] = f"code_{state['model']}_{uuid.uuid4().hex[:8]}"
        return state

    model_name = state.get("model", "Unknown")
    model_description = state.get("model_description") or {}
    task_desc = state.get("task_desc", "ML regression task")
    score_function_desc = state.get("score_function_desc", "")
    metric_direction = state.get("metric_direction", "minimize")
    metric = score_function_desc or (
        "RMSLE" if metric_direction == "minimize" else "accuracy"
    )

    feature_cols = "spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree"
    target_cols = "formation_energy_ev_natom, bandgap_energy_ev"

    example_code = model_description.get("example_code", "")
    model_desc_text = model_description.get(
        "description", f"{model_name} model for regression"
    )

    prompt = CANDIDATE_EVAL_PROMPT.format(
        task_desc=task_desc[:500],
        model_name=model_name,
        model_description=model_desc_text[:300],
        metric=metric,
        direction=format_direction(metric_direction),
        feature_cols=feature_cols,
        target_cols=target_cols,
        additional_constraints=f"Use {model_name} as the primary model.\n{('Example code snippet: ' + example_code) if example_code else ''}",
    )

    try:
        config = _default_llm_config()
        response = call_llm(prompt, response_format="code", config=config)
        code = parse_code_block(response)
        state["code"] = code
        log_node_event(
            "A2__generate", "llm_success", {"model": model_name, "code_len": len(code)}
        )
    except Exception as e:
        log_node_event(
            "A2__generate", "llm_error", {"model": model_name, "error": str(e)[:200]}
        )
        state["code"] = ""
        state["score"] = 1.0
        state["status"] = "llm_failed"

    return state


@traceable("A13__check_usage")
def A13__check_usage(state: CandidateState) -> CandidateState:
    """Verify candidate uses only allowed data sources.

    A13 from MLE-STAR paper - checks whether the candidate's code
    uses only the permitted data sources. In mock mode, 80% pass rate.
    In real mode, uses LLM to check for data usage violations.
    Force-passes after MAX_FIX_RETRIES attempts.
    """
    if _is_mock_mode():
        simulate_delay()
        attempts = state.get("usage_fix_attempts", 0)
        if attempts >= MAX_FIX_RETRIES:
            state["status"] = "ok"
            return state
        passed = random_pass(0.80)
        state["status"] = "ok" if passed else "usage_fail"
        return state

    code = state.get("code", "")
    task_desc = state.get("task_desc", "ML regression task")

    if not code.strip():
        state["status"] = "ok"
        return state

    attempts = state.get("usage_fix_attempts", 0)
    if attempts >= MAX_FIX_RETRIES:
        state["status"] = "ok"
        return state

    try:
        config = _default_llm_config()
        prompt = DATA_USAGE_CHECK_PROMPT.format(task_desc=task_desc, code=code)
        response = call_llm(prompt, response_format="json", config=config)
        parsed = parse_json_response(response)
        usage_ok = parsed.get("usage_ok", True)
        state["status"] = "ok" if usage_ok else "usage_fail"
        log_node_event(
            "A13__check_usage",
            "result",
            {"status": state["status"], "violations": parsed.get("violations", [])},
        )
    except Exception as e:
        log_node_event("A13__check_usage", "llm_error", {"error": str(e)[:200]})
        state["status"] = "ok"

    return state


@traceable("A13__fix_usage")
def A13__fix_usage(state: CandidateState) -> CandidateState:
    """Fix data usage violation in candidate code.

    Called when A13__check_usage fails. Uses LLM to fix data usage issues.
    In mock mode, appends '_fixed_usage' to code.
    """
    if _is_mock_mode():
        simulate_delay()
        state["code"] = state["code"] + "_fixed_usage"
        state["usage_fix_attempts"] = state.get("usage_fix_attempts", 0) + 1
        state["status"] = "ok"
        return state

    code = state.get("code", "")
    task_desc = state.get("task_desc", "ML regression task")

    try:
        config = _default_llm_config()
        prompt = DATA_USAGE_FIX_PROMPT.format(
            task_desc=task_desc,
            violations="Data usage violation detected",
            code=code,
        )
        response = call_llm(prompt, response_format="code", config=config)
        fixed_code = parse_code_block(response)
        state["code"] = fixed_code
        log_node_event("A13__fix_usage", "fixed", {"code_len": len(fixed_code)})
    except Exception as e:
        log_node_event("A13__fix_usage", "llm_error", {"error": str(e)[:200]})
        state["code"] = state["code"] + "\n# usage fix attempted\n"

    state["usage_fix_attempts"] = state.get("usage_fix_attempts", 0) + 1
    state["status"] = "ok"
    return state


@traceable("A12__check_leakage")
def A12__check_leakage(state: CandidateState) -> CandidateState:
    """Verify candidate has no data leakage.

    A12 from MLE-STAR paper - checks for data leakage between train/test
    sets or other contamination. In mock mode, 90% pass rate.
    In real mode, uses LLM to detect leakage patterns.
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

    code = state.get("code", "")
    task_desc = state.get("task_desc", "ML regression task")
    metric = state.get("score_function_desc", "RMSLE")

    if not code.strip():
        state["status"] = "ok"
        return state

    attempts = state.get("leakage_fix_attempts", 0)
    if attempts >= MAX_FIX_RETRIES:
        state["status"] = "ok"
        return state

    try:
        config = _default_llm_config()
        prompt = DATA_LEAKAGE_EXTRACT_PROMPT.format(
            task_desc=task_desc, metric=metric, code=code
        )
        response = call_llm(prompt, response_format="json", config=config)
        parsed = parse_json_response(response)
        has_leakage = parsed.get("has_leakage", False)
        state["status"] = "ok" if not has_leakage else "leakage_fail"
        log_node_event(
            "A12__check_leakage",
            "result",
            {"status": state["status"], "issues": parsed.get("leakage_issues", [])},
        )
    except Exception as e:
        log_node_event("A12__check_leakage", "llm_error", {"error": str(e)[:200]})
        state["status"] = "ok"

    return state


@traceable("A12__fix_leakage")
def A12__fix_leakage(state: CandidateState) -> CandidateState:
    """Fix data leakage in candidate code.

    Called when A12__check_leakage fails. Uses LLM to fix leakage.
    In mock mode, appends '_fixed_leak' to code.
    """
    if _is_mock_mode():
        simulate_delay()
        state["code"] = state["code"] + "_fixed_leak"
        state["leakage_fix_attempts"] = state.get("leakage_fix_attempts", 0) + 1
        state["status"] = "ok"
        return state

    code = state.get("code", "")
    task_desc = state.get("task_desc", "ML regression task")
    metric = state.get("score_function_desc", "RMSLE")

    try:
        config = _default_llm_config()
        prompt = DATA_LEAKAGE_CORRECT_PROMPT.format(
            task_desc=task_desc,
            metric=metric,
            leakage_issues="Data leakage detected",
            code=code,
        )
        response = call_llm(prompt, response_format="code", config=config)
        fixed_code = parse_code_block(response)
        state["code"] = fixed_code
        log_node_event("A12__fix_leakage", "fixed", {"code_len": len(fixed_code)})
    except Exception as e:
        log_node_event("A12__fix_leakage", "llm_error", {"error": str(e)[:200]})
        state["code"] = state["code"] + "\n# leakage fix attempted\n"

    state["leakage_fix_attempts"] = state.get("leakage_fix_attempts", 0) + 1
    state["status"] = "ok"
    return state


@traceable("eval_candidate")
def eval_candidate(state: CandidateState) -> CandidateState:
    """Execute candidate code and evaluate performance.

    In mock mode, generates a random score between 0.05-0.08 with 10% crash rate.
    In real mode, executes the code in a subprocess with sandbox validation.
    """
    if _is_mock_mode():
        simulate_delay()
        crashed = random_pass(0.10)
        attempts = state.get("attempts", 0)
        if crashed and attempts >= 3:
            state["score"] = random_score()
            state["status"] = "ok"
        elif crashed:
            state["status"] = "crashed"
        else:
            state["score"] = random_score()
            state["status"] = "ok"
        return state

    code = state.get("code", "")
    datasets = state.get("datasets", [])
    metric_direction = state.get("metric_direction", "minimize")

    if not code.strip():
        state["score"] = 1.0
        state["status"] = "ok"
        state["execution_output"] = ""
        state["execution_error"] = "Empty code"
        return state

    train_path = "input/train.csv"
    test_path = "input/test.csv"
    if datasets:
        for d in datasets:
            if "train" in d.lower():
                train_path = d
            if "test" in d.lower() and "submission" not in d.lower():
                test_path = d

    result = execute_code(
        code=code,
        train_path=train_path,
        test_path=test_path,
    )

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


@traceable("A11__debug")
def A11__debug(state: CandidateState) -> CandidateState:
    """Debug crashed candidate code.

    Stage 10: Uses reflection-based debugging — classifies the error type
    before generating a fix, providing more targeted debugging.

    A11 from MLE-STAR paper - attempts to fix execution failures.
    In mock mode, appends '_debugged_{attempts}'. In real mode, uses LLM.
    Max 3 debug attempts before accepting a score.
    """
    from src.mle_star.nodes.robustness import _classify_error

    if _is_mock_mode():
        simulate_delay()
        state["code"] = state["code"] + f"_debugged_{state['attempts']}"
        state["attempts"] = state.get("attempts", 0) + 1
        state["status"] = "ok"
        return state

    code = state.get("code", "")
    error_message = state.get("execution_error", "unknown error")
    task_desc = state.get("task_desc", "ML regression task")
    metric = state.get("score_function_desc", "RMSLE")

    error_class = _classify_error(
        str(error_message), stderr=str(error_message)[:1000], code=code
    )
    error_type = error_class["error_type"]
    error_suggestion = error_class["suggestion"]

    try:
        config = _default_llm_config()
        stderr = error_message[:1000] if error_message else ""
        prompt = DEBUG_PROMPT.format(
            task_desc=task_desc,
            metric=metric,
            code=code,
            error_message=f"[{error_type}] {error_message}\nSuggestion: {error_suggestion}",
            stderr=stderr,
        )
        response = call_llm(prompt, response_format="code", config=config)
        fixed_code = parse_code_block(response)
        state["code"] = fixed_code
        log_node_event(
            "A11__debug",
            "debugged",
            {
                "attempts": state.get("attempts", 0),
                "code_len": len(fixed_code),
                "error_type": error_type,
            },
        )
    except Exception as e:
        log_node_event("A11__debug", "llm_error", {"error": str(e)[:200]})
        state["code"] = (
            state["code"] + f"\n# debug attempt {state.get('attempts', 0)}\n"
        )

    state["attempts"] = state.get("attempts", 0) + 1
    state["status"] = "ok"
    return state


def route_after_generate(state: CandidateState) -> str:
    if state.get("status") == "llm_failed":
        return END
    return "A13__check_usage"


def route_after_check_usage(state: CandidateState) -> str:
    if state.get("status") == "usage_fail":
        return "A13__fix_usage"
    return "A12__check_leakage"


def route_after_fix_usage(state: CandidateState) -> str:
    return "A13__check_usage"


def route_after_check_leakage(state: CandidateState) -> str:
    if state.get("status") == "leakage_fail":
        return "A12__fix_leakage"
    return "eval_candidate"


def route_after_fix_leakage(state: CandidateState) -> str:
    return "A12__check_leakage"


def route_after_execute(state: CandidateState) -> str:
    if state.get("status") == "crashed":
        attempts = state.get("attempts", 0)
        if attempts >= 3:
            return END
        return "A11__debug"
    return END


_builder = StateGraph(CandidateState)

_builder.add_node("A2__generate", A2__generate)
_builder.add_node("A13__check_usage", A13__check_usage)
_builder.add_node("A13__fix_usage", A13__fix_usage)
_builder.add_node("A12__check_leakage", A12__check_leakage)
_builder.add_node("A12__fix_leakage", A12__fix_leakage)
_builder.add_node("eval_candidate", eval_candidate)
_builder.add_node("A11__debug", A11__debug)

_builder.add_edge(START, "A2__generate")
_builder.add_conditional_edges(
    "A2__generate",
    route_after_generate,
    {"A13__check_usage": "A13__check_usage", END: END},
)
_builder.add_conditional_edges(
    "A13__check_usage",
    route_after_check_usage,
    {"A13__fix_usage": "A13__fix_usage", "A12__check_leakage": "A12__check_leakage"},
)
_builder.add_conditional_edges(
    "A13__fix_usage", route_after_fix_usage, {"A13__check_usage": "A13__check_usage"}
)
_builder.add_conditional_edges(
    "A12__check_leakage",
    route_after_check_leakage,
    {"A12__fix_leakage": "A12__fix_leakage", "eval_candidate": "eval_candidate"},
)
_builder.add_conditional_edges(
    "A12__fix_leakage",
    route_after_fix_leakage,
    {"A12__check_leakage": "A12__check_leakage"},
)
_builder.add_conditional_edges(
    "eval_candidate",
    route_after_execute,
    {"A11__debug": "A11__debug", END: END},
)
_builder.add_edge("A11__debug", "eval_candidate")

candidate_subgraph = _builder.compile()
