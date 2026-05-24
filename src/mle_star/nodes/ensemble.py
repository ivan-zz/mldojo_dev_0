"""Ensemble phase nodes: A9__plan_ensemble, A10__implement_ensemble.

A9 generates an ensemble strategy plan based on input solutions and scores.
A10 implements the plan as merged ensemble code.

Mock mode is controlled by _is_mock_mode() which checks MOCK_MODE config
and the MLE_MOCK_MODE environment variable. When mock mode is active,
deterministic placeholder behavior is used. In real mode, LLM calls drive
planning and implementation.
"""

import os

from src.mle_star.state.shared import (
    traceable,
    simulate_delay,
    log_node_event,
    normalize_score,
    call_llm,
    _default_llm_config,
    LLMConfig,
    parse_code_block,
    format_direction,
)
from src.mle_star.config import MOCK_MODE
from src.mle_star.prompts.ensemble import ENSEMBLE_PLANNER_PROMPT, ENSEMBLER_PROMPT


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


@traceable("A9__plan_ensemble")
def A9__plan_ensemble(state: dict) -> dict:
    """Generate an ensemble strategy plan.

    A9 from MLE-STAR paper — analyzes the input solutions and their scores,
    then proposes an ensemble strategy (e.g., weighted average, stacking).

    Mock: returns a plan string based on the number of solutions and scores.
    Real: uses LLM to generate an ensemble plan.
    """
    if _is_mock_mode():
        return _A9__plan_ensemble_mock(state)
    return _A9__plan_ensemble_real(state)


def _A9__plan_ensemble_mock(state: dict) -> dict:
    simulate_delay()

    num_solutions = len(state.get("ensemble_solutions", []))
    scores = state.get("ensemble_input_scores", [])
    ensemble_round = state.get("ensemble_round", 0)

    if scores:
        metric_direction = state.get("metric_direction", "maximize")
        best_score = max(scores, key=lambda s: normalize_score(s, metric_direction))
        strategy = f"Weighted average of {num_solutions} models (best={best_score:.4f})"
    else:
        strategy = f"Single model ensemble (round {ensemble_round})"

    plan = f"Ensemble plan r{ensemble_round}: {strategy}"

    log_node_event(
        "A9__plan_ensemble",
        "output",
        {
            "plan": plan[:80],
            "num_solutions": num_solutions,
            "ensemble_round": ensemble_round,
            "status": "planned",
        },
    )

    return {
        "current_ensemble_plan": plan,
        "status": "planned",
    }


def _A9__plan_ensemble_real(state: dict) -> dict:
    ensemble_solutions = state.get("ensemble_solutions", [])
    ensemble_input_scores = state.get("ensemble_input_scores", [])
    ensemble_round = state.get("ensemble_round", 0)
    best_ensemble_score = state.get("best_ensemble_score", None)
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "score")
    metric_direction = state.get("metric_direction", "maximize")

    score_descriptions = ", ".join(
        f"Solution {i + 1}: {metric}={s:.4f}"
        for i, s in enumerate(ensemble_input_scores)
    )

    ensemble_plans = state.get("ensemble_plans", [])
    if ensemble_plans:
        previous_plans = "\n".join(
            f"- Round {i}: {p}" for i, p in enumerate(ensemble_plans)
        )
    else:
        previous_plans = "No previous ensemble plans."

    best_score_str = (
        f"{best_ensemble_score:.4f}" if best_ensemble_score is not None else "N/A"
    )

    prompt = ENSEMBLE_PLANNER_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        direction=format_direction(metric_direction),
        score_descriptions=score_descriptions,
        previous_plans=previous_plans,
        best_ensemble_score=best_score_str,
        ensemble_round=ensemble_round,
    )

    try:
        plan = call_llm(prompt, config=_default_llm_config())
    except Exception as e:
        log_node_event(
            "A9__plan_ensemble",
            "llm_failed",
            {"error": str(e)[:300]},
        )
        return _A9__plan_ensemble_mock(state)

    log_node_event(
        "A9__plan_ensemble",
        "output",
        {
            "plan": str(plan)[:80],
            "num_solutions": len(ensemble_solutions),
            "ensemble_round": ensemble_round,
            "status": "planned",
        },
    )

    return {
        "current_ensemble_plan": plan,
        "status": "planned",
    }


@traceable("A10__implement_ensemble")
def A10__implement_ensemble(state: dict) -> dict:
    """Implement the ensemble plan as merged code.

    A10 from MLE-STAR paper — takes the ensemble plan and input solutions,
    produces merged ensemble code that combines predictions.

    Mock: concatenates the best solution with an ensemble comment block.
    Real: uses LLM to generate ensemble implementation code.
    """
    if _is_mock_mode():
        return _A10__implement_ensemble_mock(state)
    return _A10__implement_ensemble_real(state)


def _A10__implement_ensemble_mock(state: dict) -> dict:
    simulate_delay()

    ensemble_plan = state.get("current_ensemble_plan", "")
    ensemble_solutions = state.get("ensemble_solutions", [""])
    best_ensemble_code = state.get(
        "best_ensemble_code", ensemble_solutions[0] if ensemble_solutions else ""
    )
    ensemble_round = state.get("ensemble_round", 0)

    current_code = best_ensemble_code or (
        ensemble_solutions[0] if ensemble_solutions else ""
    )

    ensemble_code = current_code + f"\n# ensemble_v{ensemble_round}\n"

    log_node_event(
        "A10__implement_ensemble",
        "output",
        {
            "plan": ensemble_plan[:60],
            "code_len": len(ensemble_code),
            "ensemble_round": ensemble_round,
            "status": "implemented",
        },
    )

    return {
        "current_ensemble_code": ensemble_code,
        "status": "implemented",
    }


def _A10__implement_ensemble_real(state: dict) -> dict:
    ensemble_plan = state.get("current_ensemble_plan", "")
    ensemble_solutions = state.get("ensemble_solutions", [""])
    ensemble_input_scores = state.get("ensemble_input_scores", [])
    best_ensemble_code = state.get("best_ensemble_code", None)
    ensemble_round = state.get("ensemble_round", 0)
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "score")
    metric_direction = state.get("metric_direction", "maximize")

    solutions_with_scores = "\n\n".join(
        f"--- Solution {i + 1} (score={s:.4f}) ---\n{code}"
        for i, (code, s) in enumerate(
            zip(
                ensemble_solutions,
                ensemble_input_scores or [0.0] * len(ensemble_solutions),
            )
        )
    )

    if best_ensemble_code:
        best_ensemble_section = f"CURRENT BEST ENSEMBLE CODE (to improve upon):\n```python\n{best_ensemble_code}\n```"
    else:
        best_ensemble_section = ""

    prompt = ENSEMBLER_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        direction=format_direction(metric_direction),
        plan=ensemble_plan,
        solutions_with_scores=solutions_with_scores,
        best_ensemble_section=best_ensemble_section,
    )

    try:
        base_config = _default_llm_config()
        ensemble_config = LLMConfig(
            provider=base_config.provider,
            model=base_config.model,
            base_url=base_config.base_url,
            api_key=base_config.api_key,
            temperature=base_config.temperature,
            max_tokens=16384,
            timeout=base_config.timeout,
        )
        raw_response = call_llm(prompt, config=ensemble_config)
        code = parse_code_block(raw_response)
    except Exception as e:
        log_node_event(
            "A10__implement_ensemble",
            "llm_failed",
            {"error": str(e)[:300]},
        )
        fallback_code = best_ensemble_code or (
            ensemble_solutions[0] if ensemble_solutions else ""
        )
        fallback_code = (
            fallback_code or ""
        ) + f"\n# ensemble_v{ensemble_round}_llm_fallback\n"
        log_node_event(
            "A10__implement_ensemble",
            "output",
            {
                "plan": str(ensemble_plan)[:60],
                "code_len": len(fallback_code),
                "ensemble_round": ensemble_round,
                "status": "llm_failed",
            },
        )
        return {
            "current_ensemble_code": fallback_code,
            "status": "llm_failed",
        }

    log_node_event(
        "A10__implement_ensemble",
        "output",
        {
            "plan": str(ensemble_plan)[:60],
            "code_len": len(code),
            "ensemble_round": ensemble_round,
            "status": "implemented",
        },
    )

    return {
        "current_ensemble_code": code,
        "status": "implemented",
    }
