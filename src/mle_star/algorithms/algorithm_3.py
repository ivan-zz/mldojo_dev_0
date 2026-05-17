"""Algorithm 3: Ensemble (Paper Algorithm 3).

Orchestrates the ensemble round loop via a Python-level `for r in range(R)`
iteration, matching the pattern used by algorithm_2.py. Each round is
wrapped in a SubgraphSpan for grouped Langfuse traces.

Architecture:
    Layer 1: algorithm_3.run_algorithm3() — Python for loop, SubgraphSpan per round
    Layer 2: ensemble_round_subgraph — single-pass A9→A10→A12→eval per round
              with debug retry and leakage fix conditional edges

The round count R is read from config.py (MAX_ENSEMBLE_ROUNDS env var).

Tracing hierarchy:
    Algorithm3
    ├── ensemble_round_r0 (SubgraphSpan)
    │   └── EnsembleRoundSubgraph
    ├── ensemble_round_r1 (SubgraphSpan)
    │   └── EnsembleRoundSubgraph
    └── ...
"""

import os
from typing import Dict

from src.mle_star.state.alg3_state import Alg3State, EnsembleRoundState
from src.mle_star.state.shared import (
    MLELogger,
    SubgraphSpan,
    _current_mle_logger,
    _current_phase,
    _current_run_dir,
    _current_run_id,
    generate_run_dir,
    get_checkpointer,
    get_run_dir,
    get_thread_id,
    log_node_event,
    normalize_score,
    propagate_attributes,
    traceable,
    langfuse,
    reset_llm_call_counter,
)
from src.mle_star.config import MAX_ENSEMBLE_ROUNDS


def _alg3_to_round_state(
    state: Alg3State, r: int, best_code: str, best_score: float
) -> dict:
    """Map Alg3State to EnsembleRoundState for a single round invocation.

    The Python-level loop in run_algorithm3() manages cumulative state
    (ensemble_scores, ensemble_plans, best tracking) across rounds.
    The round subgraph only sees fields for one pass.
    """
    return {
        "ensemble_solutions": state.get("ensemble_solutions", []),
        "ensemble_input_scores": state.get("ensemble_input_scores", []),
        "metric_direction": state.get("metric_direction", "maximize"),
        "current_ensemble_plan": state.get("current_ensemble_plan", ""),
        "current_ensemble_code": state.get("current_ensemble_code", best_code),
        "current_ensemble_score": state.get("current_ensemble_score", best_score),
        "ensemble_round": r,
        "execution_output": "",
        "execution_error": None,
        "execution_score": None,
        "debug_retries": 0,
        "leakage_status": None,
        "leakage_code_block": None,
        "best_ensemble_code": best_code,
        "best_ensemble_score": best_score,
        "status": "start",
    }


def _round_result_to_alg3(result: dict, state: Alg3State) -> dict:
    """Map EnsembleRoundState result back to Alg3State updates.

    Only updates fields that change within a round. Cumulative fields
    (ensemble_scores, ensemble_plans) are managed by the outer loop.
    """
    return {
        "current_ensemble_plan": result.get("current_ensemble_plan", ""),
        "current_ensemble_code": result.get("current_ensemble_code", ""),
        "current_ensemble_score": result.get("current_ensemble_score", 0),
        "best_ensemble_code": result.get("best_ensemble_code", ""),
        "best_ensemble_score": result.get("best_ensemble_score", 0),
        "leakage_status": result.get("leakage_status"),
        "leakage_code_block": result.get("leakage_code_block"),
        "status": result.get("status", ""),
    }


def run_algorithm3(state: Alg3State) -> Dict:
    """Run the full Algorithm 3 ensemble round loop.

    Python-level for loop over R ensemble rounds. Each round:
    1. Maps Alg3State to EnsembleRoundState
    2. Invokes the ensemble round subgraph inside a SubgraphSpan
    3. Maps result back and tracks best ensemble code/score

    Returns updated state with best_ensemble_code, best_ensemble_score, etc.
    """
    R = MAX_ENSEMBLE_ROUNDS
    reset_llm_call_counter()
    ensemble_solutions = state.get("ensemble_solutions", [])
    ensemble_input_scores = state.get("ensemble_input_scores", [])
    metric_direction = state.get("metric_direction", "maximize")
    current_best_score = (
        max(ensemble_input_scores, key=lambda s: normalize_score(s, metric_direction))
        if ensemble_input_scores
        else 0
    )
    best_ensemble_code = ensemble_solutions[0] if ensemble_solutions else ""
    best_ensemble_score = current_best_score
    stage_history = list(state.get("stage_history", []))

    ensemble_scores = list(state.get("ensemble_scores", []))
    ensemble_plans = list(state.get("ensemble_plans", []))

    from src.mle_star.subgraphs.ensemble_round_subgraph import (
        get_ensemble_round_subgraph,
    )

    round_subgraph = get_ensemble_round_subgraph()

    for r in range(R):
        with SubgraphSpan(
            f"ensemble_round_r{r}",
            input_data={
                "round": r,
                "best_score": best_ensemble_score,
                "num_solutions": len(ensemble_solutions),
                "status": "start_round",
            },
        ) as round_span:
            round_state: EnsembleRoundState = _alg3_to_round_state(
                state, r, best_ensemble_code, best_ensemble_score
            )

            round_config = {"configurable": {"thread_id": f"alg3_r{r}"}}
            result = round_subgraph.invoke(round_state, round_config)

            round_best_code = result.get("best_ensemble_code", best_ensemble_code)
            round_best_score = result.get("best_ensemble_score", best_ensemble_score)

            if (
                normalize_score(round_best_score, metric_direction)
                > normalize_score(best_ensemble_score, metric_direction)
                and round_best_code
            ):
                best_ensemble_score = round_best_score
                best_ensemble_code = round_best_code

            round_score = result.get("current_ensemble_score", 0)
            round_plan = result.get("current_ensemble_plan", "")
            ensemble_scores.append(round_score)
            ensemble_plans.append(round_plan)

            round_span.set_output(
                {
                    "round": r,
                    "round_score": round_score,
                    "best_ensemble_score": best_ensemble_score,
                    "status": result.get("status", ""),
                }
            )

        log_node_event(
            "Algorithm3",
            "round_complete",
            {
                "round": r,
                "round_score": round_score,
                "best_ensemble_score": best_ensemble_score,
            },
        )

    return {
        "best_ensemble_code": best_ensemble_code,
        "best_ensemble_score": best_ensemble_score,
        "ensemble_scores": ensemble_scores,
        "ensemble_plans": ensemble_plans,
        "ensemble_round": R,
        "stage_history": stage_history,
        "status": "done",
    }


def run(
    initial_state: Dict = None,
    run_dir: str = None,
    thread_id: str = None,
    session_id: str = None,
):
    """Run Algorithm 3 with the given configuration.

    Wraps the entire ensemble execution in a root SubgraphSpan so all
    round traces appear grouped under one "Algorithm3" trace in Langfuse.

    Args:
        initial_state: Optional initial state dict.
        run_dir: Optional run directory path.
        thread_id: Optional thread ID.
        session_id: Optional session ID for Langfuse session grouping.

    Returns:
        Final state after algorithm execution.
    """
    if run_dir is None:
        run_dir = generate_run_dir()

    os.makedirs(run_dir, exist_ok=True)
    run_id = os.path.basename(run_dir.rstrip("/"))

    run_dir_token = _current_run_dir.set(run_dir)
    run_id_token = _current_run_id.set(run_id)
    phase_token = _current_phase.set("ensemble")

    mle_logger = MLELogger(run_dir=run_dir)
    logger_token = _current_mle_logger.set(mle_logger)

    if initial_state is None:
        initial_state = {
            "ensemble_solutions": [],
            "ensemble_input_scores": [],
            "metric_direction": "maximize",
            "ensemble_plans": [],
            "ensemble_scores": [],
            "current_ensemble_plan": "",
            "current_ensemble_code": "",
            "current_ensemble_score": 0,
            "ensemble_round": 0,
            "execution_output": "",
            "execution_error": None,
            "execution_score": None,
            "debug_history": [],
            "leakage_status": None,
            "leakage_code_block": None,
            "debug_retries": 0,
            "best_ensemble_code": "",
            "best_ensemble_score": 0,
            "stage_history": [],
            "status": "start",
        }

    if thread_id is None:
        thread_id = get_thread_id(run_dir)

    if session_id is None:
        session_id = thread_id

    log_node_event(
        "Algorithm3",
        "run_start",
        {
            "num_solutions": len(initial_state.get("ensemble_solutions", [])),
            "run_dir": run_dir,
        },
    )

    result: Dict = {}
    try:
        with SubgraphSpan(
            "Algorithm3",
            input_data={
                "num_solutions": len(initial_state.get("ensemble_solutions", [])),
                "status": "start",
            },
        ) as root_span:
            with propagate_attributes(session_id=session_id):
                result = run_algorithm3(initial_state)
                root_span.set_output(
                    {
                        "status": result.get("status", ""),
                        "best_ensemble_score": result.get("best_ensemble_score", 0),
                        "ensemble_round": result.get("ensemble_round", 0),
                    }
                )
    finally:
        log_node_event(
            "Algorithm3",
            "run_end",
            {
                "status": result.get("status", ""),
                "best_ensemble_score": result.get("best_ensemble_score", 0),
            },
        )
        mle_logger.close()
        _current_run_dir.reset(run_dir_token)
        _current_run_id.reset(run_id_token)
        _current_phase.reset(phase_token)
        _current_mle_logger.reset(logger_token)
        try:
            langfuse.flush()
        except Exception:
            log_node_event(
                "Algorithm3", "flush_error", {"error": "langfuse.flush() failed"}
            )

    return result
