"""Refinement phase nodes: A7__implement, eval_refinement, A8__plan, A11__debug_refine.

A_verify and A_sast are now in nodes/verification.py (Stage 10 move).
Routing functions (route_after_verify, route_after_sast, route_after_eval_step)
remain here as they are specific to the refinement subgraph topology.

Mock implementations for Stage 3/4. Real implementations built in Stage 8.
"""

import os

from langgraph.graph import END

from src.mle_star.state.shared import (
    traceable,
    simulate_delay,
    log_node_event,
    random_score,
    random_pass,
    normalize_score,
    call_llm,
    _default_llm_config,
    LLMConfig,
    parse_code_block,
    replace_code_block,
    format_direction,
)
from src.mle_star.config import MOCK_MODE, MAX_DEBUG_RETRIES
from src.mle_star.prompts.refinement import CODER_PROMPT, PLANNER_PROMPT
from src.mle_star.prompts.robustness import DEBUG_PROMPT
from src.mle_star.nodes.execution import execute_code
from src.mle_star.nodes.robustness import _classify_error


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


@traceable("A7__implement")
def A7__implement(state: dict) -> dict:
    """Implement a refinement plan as code for the target block.

    A7 from MLE-STAR paper - takes the current plan (p_0 or p_k from A8)
    and generates the refined code block. The refined block replaces the
    target block in the full solution.

    Mock: returns a mock refined code block based on the plan.
    Real: uses LLM (CODER_PROMPT) to generate the refined block.
    """
    if _is_mock_mode():
        simulate_delay()

        current_plan = state.get("current_plan", state.get("initial_plan", ""))
        target_block = state.get("target_block", "")
        current_solution = state.get("current_solution", "")

        refined_code = target_block + "# refined\n"
        candidate_solution = _replace_block_mock(
            current_solution, target_block, refined_code
        )

        log_node_event(
            "A7__implement",
            "output",
            {"plan": current_plan[:60], "refined_code_len": len(refined_code)},
        )

        return {
            "refined_code": refined_code,
            "candidate_solution": candidate_solution,
            "status": "implemented",
        }

    current_plan = state.get("current_plan", state.get("initial_plan", ""))
    target_block = state.get("target_block", "")
    current_solution = state.get("current_solution", "")
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "accuracy")
    metric_direction = state.get("metric_direction", "maximize")

    previous_attempts = state.get("previous_attempts", "")
    if isinstance(previous_attempts, list):
        previous_attempts = "\n".join(str(a) for a in previous_attempts)

    prompt = CODER_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        direction=format_direction(metric_direction),
        current_solution=current_solution,
        target_block=target_block,
        current_plan=current_plan,
        previous_attempts=previous_attempts,
    )

    try:
        base_config = _default_llm_config()
        code_config = LLMConfig(
            provider=base_config.provider,
            model=base_config.model,
            base_url=base_config.base_url,
            api_key=base_config.api_key,
            temperature=base_config.temperature,
            max_tokens=8192,
            timeout=base_config.timeout,
        )
        response = call_llm(prompt, response_format="code", config=code_config)
        refined_code = parse_code_block(response)
    except Exception as e:
        log_node_event(
            "A7__implement",
            "llm_failed",
            {"error": str(e)[:200]},
        )
        refined_code = target_block + "# llm_failed\n"

    candidate_solution = replace_code_block(
        current_solution, target_block, refined_code
    )

    if candidate_solution == current_solution:
        log_node_event(
            "A7__implement",
            "replace_failed_unchanged",
            {"target_block_preview": target_block[:80]},
        )

    log_node_event(
        "A7__implement",
        "output",
        {"plan": current_plan[:60], "refined_code_len": len(refined_code)},
    )

    return {
        "refined_code": refined_code,
        "candidate_solution": candidate_solution,
        "status": "implemented",
    }


from src.mle_star.nodes.verification import A_verify, A_sast


@traceable("eval_refinement")
def eval_refinement(state: dict) -> dict:
    """Execute the candidate solution and evaluate performance.

    Substitutes the refined block into the full solution and runs it.
    If execution fails and debug retries are available, routes to debug.

    Mock: generates a random score, 10% chance of error.
    Real: executes the code with execute_code() and parses the score.
    """
    if _is_mock_mode():
        simulate_delay()

        candidate_solution = state.get("candidate_solution", "")
        best_score = state.get("best_score", 0.85)
        debug_retries = state.get("debug_retries", 0)
        inner_step = state.get("inner_step", 0)

        crashed = random_pass(0.10)

        if crashed and debug_retries < MAX_DEBUG_RETRIES:
            log_node_event(
                "eval_refinement",
                "output",
                {"status": "error", "debug_retries": debug_retries},
            )
            return {
                "execution_error": "Mock execution error",
                "execution_score": None,
                "status": "error",
            }

        improvement = random_score(0.001, 0.02)
        metric_direction = state.get("metric_direction", "maximize")
        score = (
            best_score + improvement
            if metric_direction == "maximize"
            else best_score - improvement
        )

        current_improved_score = state.get("improved_score", 0)
        current_improved_solution = state.get("improved_solution", "")
        improved_score = (
            score
            if normalize_score(score, metric_direction)
            > normalize_score(current_improved_score, metric_direction)
            else current_improved_score
        )
        improved_solution = (
            (candidate_solution or current_improved_solution)
            if normalize_score(score, metric_direction)
            > normalize_score(current_improved_score, metric_direction)
            else current_improved_solution
        )

        log_node_event(
            "eval_refinement",
            "output",
            {
                "status": "ok",
                "score": round(score, 4),
                "inner_step": inner_step,
            },
        )

        return {
            "execution_output": f"Final Validation Performance: {score:.4f}",
            "execution_error": None,
            "execution_score": round(score, 4),
            "improved_score": round(improved_score, 4),
            "improved_solution": improved_solution,
            "status": "ok",
        }

    candidate_solution = state.get("candidate_solution", "")
    best_score = state.get("best_score", 0.85)
    debug_retries = state.get("debug_retries", 0)
    inner_step = state.get("inner_step", 0)
    metric_direction = state.get("metric_direction", "maximize")

    result = execute_code(candidate_solution)

    if result["status"] != "ok":
        log_node_event(
            "eval_refinement",
            "output",
            {
                "status": "error",
                "debug_retries": debug_retries,
                "exit_code": result.get("exit_code"),
                "stderr_len": len(result.get("stderr", "")),
            },
        )

        stderr = result.get("stderr", "")
        stdout = result.get("stdout", "")
        error_msg = stderr or stdout or "Execution failed"

        return {
            "execution_output": stdout,
            "execution_error": error_msg,
            "execution_score": None,
            "execution_exit_code": result.get("exit_code", -1),
            "debug_retries": debug_retries,
            "status": "error",
        }

    score = result["score"]

    current_improved_score = state.get("improved_score", 0)
    current_improved_solution = state.get("improved_solution", "")

    if score is not None and normalize_score(score, metric_direction) > normalize_score(
        current_improved_score, metric_direction
    ):
        improved_score = score
        improved_solution = candidate_solution
    else:
        improved_score = current_improved_score
        improved_solution = current_improved_solution

    log_node_event(
        "eval_refinement",
        "output",
        {
            "status": "ok",
            "score": round(score, 4) if score is not None else None,
            "inner_step": inner_step,
        },
    )

    return {
        "execution_output": result.get("stdout", ""),
        "execution_error": None,
        "execution_score": round(score, 4) if score is not None else None,
        "improved_score": round(improved_score, 4),
        "improved_solution": improved_solution,
        "status": "ok",
    }


@traceable("A8__plan")
def A8__plan(state: dict) -> dict:
    """Propose next refinement plan based on previous results.

    A8 from MLE-STAR paper - analyzes previous refinement attempts
    and proposes a new plan p_{k+1}.

    Mock: generates a variant plan.
    Real: uses LLM (PLANNER_PROMPT) to generate the next plan.
    """
    if _is_mock_mode():
        simulate_delay()

        target_block = state.get("target_block", "")
        inner_step = state.get("inner_step", 0)
        previous_score = state.get("execution_score", 0)

        plan = (
            f"Plan {inner_step + 1}: Try alternative approach for "
            f"the target block (previous score={previous_score:.4f})."
        )

        log_node_event(
            "A8__plan",
            "output",
            {"plan": plan[:60], "inner_step": inner_step},
        )

        return {
            "current_plan": plan,
            "inner_step": inner_step + 1,
        }

    target_block = state.get("target_block", "")
    inner_step = state.get("inner_step", 0)
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "accuracy")
    best_score = state.get("best_score", 0)
    execution_output = state.get("execution_output", "")

    current_plans = state.get("current_plans", [])
    if isinstance(current_plans, list):
        plan_history = "\n".join(str(p) for p in current_plans)
    else:
        plan_history = str(current_plans) if current_plans else "No previous plans."

    current_scores = state.get("current_scores", [])
    if isinstance(current_scores, list):
        score_history = ", ".join(str(s) for s in current_scores)
    else:
        score_history = str(current_scores) if current_scores else "No previous scores."

    plan_history_full = f"Plans:\n{plan_history}\n\nScores:\n{score_history}"

    prompt = PLANNER_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        target_block=target_block,
        best_score=best_score,
        plan_history=plan_history_full,
        execution_output=execution_output,
    )

    try:
        plan = call_llm(prompt, response_format="text")
    except Exception as e:
        log_node_event(
            "A8__plan",
            "llm_failed",
            {"error": str(e)[:200]},
        )
        plan = (
            f"Plan {inner_step + 1}: Try alternative approach for "
            f"the target block (previous score={state.get('execution_score', 0):.4f})."
        )

    log_node_event(
        "A8__plan",
        "output",
        {"plan": plan[:60], "inner_step": inner_step},
    )

    return {
        "current_plan": plan,
        "inner_step": inner_step + 1,
    }


@traceable("A11__debug_refine")
def A11__debug_refine(state: dict) -> dict:
    """Debug a failed refinement execution.

    Stage 10: Uses reflection-based debugging — classifies the error type
    before generating a fix, providing more targeted debugging.

    Attempts to fix the error in the refined code. Increments debug_retries.
    Max MAX_DEBUG_RETRIES attempts before giving up.

    Mock: marks as fixed by appending a debug note.
    Real: uses LLM (DEBUG_PROMPT) to fix the error.
    """
    if _is_mock_mode():
        simulate_delay()

        debug_retries = state.get("debug_retries", 0) + 1
        execution_error = state.get("execution_error", "unknown error")

        error_class = _classify_error(str(execution_error))

        refined_code = (
            state.get("refined_code", "") + f"# debug attempt {debug_retries}\n"
        )
        current_solution = state.get("current_solution", "")
        target_block = state.get("target_block", "")
        candidate_solution = _replace_block_mock(
            current_solution, target_block, refined_code
        )

        log_node_event(
            "A11__debug_refine",
            "output",
            {
                "debug_retries": debug_retries,
                "error": str(execution_error)[:60],
                "error_type": error_class["error_type"],
            },
        )

        return {
            "refined_code": refined_code,
            "candidate_solution": candidate_solution,
            "debug_retries": debug_retries,
            "execution_error": None,
            "status": "debugged",
        }

    debug_retries = state.get("debug_retries", 0) + 1
    execution_error = state.get("execution_error", "unknown error")
    refined_code = state.get("refined_code", "")
    current_solution = state.get("current_solution", "")
    target_block = state.get("target_block", "")
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "accuracy")

    execution_output = state.get("execution_output", "")
    execution_error = state.get("execution_error", "unknown error")
    exit_code = state.get("execution_exit_code", -1)

    error_class = _classify_error(
        str(execution_error), stderr=execution_output, code=refined_code
    )
    error_type = error_class["error_type"]
    error_suggestion = error_class["suggestion"]

    prompt = DEBUG_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        code=refined_code,
        exit_code=exit_code,
        error_message=f"[{error_type}] {execution_error}\nSuggestion: {error_suggestion}",
        stdout_output=execution_output,
    )

    try:
        base_config_dbg = _default_llm_config()
        debug_config = LLMConfig(
            provider=base_config_dbg.provider,
            model=base_config_dbg.model,
            base_url=base_config_dbg.base_url,
            api_key=base_config_dbg.api_key,
            temperature=base_config_dbg.temperature,
            max_tokens=8192,
            timeout=base_config_dbg.timeout,
        )
        response = call_llm(prompt, response_format="code", config=debug_config)
        fixed_code = parse_code_block(response)
        candidate_solution = replace_code_block(
            current_solution, target_block, fixed_code
        )
        refined_code = fixed_code
    except Exception as e:
        log_node_event(
            "A11__debug_refine",
            "llm_failed",
            {"error": str(e)[:200]},
        )
        candidate_solution = current_solution

    log_node_event(
        "A11__debug_refine",
        "output",
        {
            "debug_retries": debug_retries,
            "error": str(execution_error)[:60],
            "error_type": error_type,
        },
    )

    return {
        "refined_code": refined_code,
        "candidate_solution": candidate_solution,
        "debug_retries": debug_retries,
        "execution_error": None,
        "status": "debugged",
    }


def route_after_verify(state: dict) -> str:
    """Route after A_verify: semantic_fail -> A7, ok -> A_sast."""
    if state.get("status") == "semantic_fail":
        return "A7__implement"
    return "A_sast"


def route_after_sast(state: dict) -> str:
    """Route after A_sast: critical -> A7, pass -> eval_refinement."""
    if state.get("status") == "critical_violation":
        return "A7__implement"
    return "eval_refinement"


def route_after_eval_step(state: dict) -> str:
    """Route after eval_refinement in the single-pass step subgraph.

    - error and debug_retries < MAX_DEBUG_RETRIES -> A11__debug_refine
    - error and debug_retries >= MAX_DEBUG_RETRIES -> END (give up)
    - ok -> END (step complete; next step handled by Python loop)
    """
    if state.get("status") == "error":
        if state.get("debug_retries", 0) < MAX_DEBUG_RETRIES:
            return "A11__debug_refine"
        return END
    return END


def _replace_block_mock(solution: str, old_block: str, new_block: str) -> str:
    """Mock code block replacement. Real version uses AST in Stage 8."""
    if not solution:
        return new_block
    if old_block and old_block in solution:
        return solution.replace(old_block, new_block, 1)
    return solution + "\n" + new_block
