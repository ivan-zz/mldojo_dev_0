"""Supervisor node for phase routing and stagnation control.

Implements rule-based routing (supervisor_decide) and LLM-driven override
(supervisor_decide_llm) for the MLE-STAR main graph.

D12 CONSTRAINT: No phase='stagnation'. No search restart.
Phase flow: search → ablation → ensemble → submission → END.
After max_full_cycles, force submission regardless.
"""

import time as _time_mod
from dataclasses import dataclass, field
from typing import Any

from src.mle_star.config import (
    MAX_OUTER_STEPS,
    MAX_INNER_STEPS,
    MAX_DEBUG_RETRIES,
    MAX_ABLATION_DEBUG_RETRIES,
    NUM_PARALLEL_SOLUTIONS,
    MAX_ENSEMBLE_ROUNDS,
    NUM_RETRIEVED_MODELS,
    EXECUTION_TIMEOUT,
)
from src.mle_star.state.system_state import MleStarSystemState
from src.mle_star.state.shared import log_node_event, normalize_score, LLMConfig


@dataclass
class SupervisorConfig:
    """Configuration for the supervisor routing logic."""

    epsilon: float = 0.001
    max_outer_steps: int = MAX_OUTER_STEPS
    max_inner_steps: int = MAX_INNER_STEPS
    max_stagnation_rounds: int = 2
    max_debug_retries: int = MAX_DEBUG_RETRIES
    max_ablation_debug_retries: int = MAX_ABLATION_DEBUG_RETRIES
    max_ensemble_rounds: int = MAX_ENSEMBLE_ROUNDS
    num_retrieved_models: int = NUM_RETRIEVED_MODELS
    num_parallel_solutions: int = NUM_PARALLEL_SOLUTIONS
    max_full_cycles: int = 2
    execution_timeout: int = EXECUTION_TIMEOUT
    llm_config: LLMConfig | None = None
    interrupt_points: list[str] = field(default_factory=list)


def supervisor_decide(
    state: MleStarSystemState | dict, config: SupervisorConfig | None = None
) -> str:
    """Rule-based routing function for the supervisor node.

    D12 CONSTRAINT: No phase='stagnation'. No search restart.
    D15: phase='search' routes to parallel_pipeline_node (L independent pipelines).
    Algorithm 3 manages its own round loop internally (R rounds via Python for loop).
    The supervisor does NOT loop back to ensemble — once Alg3 completes,
    it transitions directly to submission.
    Phases: search → (parallel pipelines) → search_done → ensemble → submission → END.
    After max_full_cycles, force submission regardless.

    Returns one of:
        'parallel_pipeline_node'  — route to parallel pipelines (D15)
        'transition_to_ensemble'  — route to ensemble (after search_done)
        'continue_ablation'       — stay in ablation (improving)
        'new_ablation_cycle'      — start new ablation cycle (stagnated)
        'transition_to_submission' — route to submission (after ensemble)
        'END'                     — pipeline complete
    """
    if config is None:
        config = SupervisorConfig()

    phase = state.get("phase", "search")
    full_cycles = state.get("full_cycles", 0)
    max_full_cycles = state.get("max_full_cycles", config.max_full_cycles)

    if full_cycles >= max_full_cycles:
        return "transition_to_submission"

    if phase == "search":
        return "parallel_pipeline_node"

    elif phase == "ablation":
        current_score = state.get("current_score") or 0
        best_score = state.get("best_score", 0)
        metric_direction = state.get("metric_direction", "maximize")

        delta_s = normalize_score(current_score, metric_direction) - normalize_score(
            best_score, metric_direction
        )

        if delta_s > config.epsilon:
            return "continue_ablation"
        elif state.get("outer_step", 0) < config.max_outer_steps - 1:
            return "new_ablation_cycle"
        else:
            return "transition_to_ensemble"

    elif phase == "ensemble":
        return "transition_to_submission"

    elif phase == "search_done":
        return "transition_to_ensemble"

    elif phase == "submission" or phase == "done":
        return "END"

    return "transition_to_ensemble"


def supervisor_node(
    state: MleStarSystemState | dict, config: SupervisorConfig | None = None
) -> dict:
    """LangGraph node wrapper for the supervisor.

    Logs the routing decision and updates phase_history.

    Note: When used as a LangGraph node, the `config` parameter receives a
    RunnableConfig, NOT a SupervisorConfig. The SupervisorConfig must be
    captured via closure (see get_mle_star_graph) or stored on state.
    If config is None (called outside LangGraph), defaults are used.
    """
    decision = supervisor_decide(state, config)

    phase_map = {
        "parallel_pipeline_node": "search",
        "transition_to_ensemble": "ensemble",
        "continue_ablation": "ablation",
        "new_ablation_cycle": "ablation",
        "transition_to_submission": "submission",
        "END": "done",
    }

    next_phase = phase_map.get(decision, state.get("phase", "search"))

    log_node_event(
        "supervisor",
        "transition",
        {
            "decision": decision,
            "from_phase": state.get("phase"),
            "to_phase": next_phase,
        },
    )

    return {
        "phase_history": [
            {
                "phase": next_phase,
                "decision": decision,
                "timestamp": _time_mod.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        ],
    }


def supervisor_decide_llm(
    state: MleStarSystemState | dict, config: SupervisorConfig | None = None
) -> str:
    """LLM-driven override for supervisor routing.

    Placeholder: falls back to rule-based logic.
    Will be implemented with LLM integration in Stage 7+.
    """
    return supervisor_decide(state, config)
