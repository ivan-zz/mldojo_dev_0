"""Algorithm 2: Ablation + Refinement (Paper Algorithm 2).

Orchestrates the outer loop for iterative solution improvement using a
Python-level `for t in range(T)` loop. Each iteration invokes the ablation
subgraph and then runs K refinement iterations via Python-level loop, each
wrapped in a SubgraphSpan for grouped Langfuse traces.

Architecture: Three-layer nested subgraph with Python-level loops
    Layer 1: algorithm_2.run() — Python for loop, SubgraphSpan per iteration
    Layer 2: ablation_subgraph — single-pass, Send API fan-out for variants
    Layer 3: refinement_step_subgraph — single-pass A7→verify→sast→eval
              invoked K times in a Python loop, each wrapped in SubgraphSpan

The outer loop AND inner loop are both Python-level (not LangGraph conditional
edges). This produces clean nested traces in Langfuse:
    Algorithm2
    ├── ablation_cycle_t0 (SubgraphSpan)
    │   ├── AblationSubgraph (A4→fan-out→A5→A6)
    │   └── RefinementSubgraph
    │       └── refinement_phase (SubgraphSpan grouping K steps)
    │           ├── refinement_step_t0_k0
    │           │   └── RefinementStepSubgraph
    │           └── refinement_step_t0_k1
    │               └── RefinementStepSubgraph
    ├── ablation_cycle_t1 (SubgraphSpan)
    │   └── ...
    └── ...

AblationSubgraph and RefinementSubgraph are siblings under ablation_cycle_t{t},
not nested. This matches the thesis subgraph design where the ablation phase
and refinement phase are logically distinct stages within each cycle.

Fan-out nodes (`candidate_flow_node`, `merge_flow_node`, `ablation_variant_flow_node`,
`pipeline_flow_node`) do **not** have `@traceable` decorators — their `SubgraphSpan`
provides the grouping level directly, avoiding redundant nesting.

Hyperparameters (MAX_OUTER_STEPS, MAX_INNER_STEPS, etc.) are read from
environment variables via config.py, falling back to defaults.
"""

import os
from typing import Dict

from src.mle_star.state.alg2_state import (
    Alg2State,
    AblationCycleState,
    RefinementInnerState,
)
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
from src.mle_star.config import MAX_OUTER_STEPS, MAX_INNER_STEPS, MAX_DEBUG_RETRIES
from src.mle_star.subgraphs.ablation_subgraph import get_ablation_subgraph
from src.mle_star.subgraphs.refinement_subgraph import get_refinement_step_subgraph


@traceable("Algorithm2")
def run_algorithm2(state: Alg2State) -> Alg2State:
    """Run the full Algorithm 2 outer loop.

    Python-level for loop over T outer iterations. Each iteration:
    1. Invokes the ablation subgraph (single-pass) inside a SubgraphSpan
    2. Runs K refinement iterations in a Python loop, each inside a SubgraphSpan
    3. Merges results: updates best_score, extends accumulated lists

    Returns updated state with improved_solution, improved_score, etc.
    """
    T = MAX_OUTER_STEPS
    K = MAX_INNER_STEPS
    reset_llm_call_counter()
    best_score = state.get("best_score", 0)
    metric_direction = state.get("metric_direction", "maximize")
    improved_solution = state.get("current_solution", "")
    improved_score = best_score
    ablation_summaries = list(state.get("ablation_summaries", []))
    refined_blocks = list(state.get("refined_blocks", []))
    debug_history = list(state.get("debug_history", []))
    security_violations = list(state.get("security_violations", []))
    stage_history = list(state.get("stage_history", []))

    ablation_subgraph = get_ablation_subgraph()
    refinement_step_subgraph = get_refinement_step_subgraph()

    for t in range(T):
        with SubgraphSpan(
            f"ablation_cycle_t{t}",
            input_data={
                "outer_step": t,
                "best_score": best_score,
                "status": "start_cycle",
            },
        ) as cycle_span:
            ablation_input: AblationCycleState = {
                "current_solution": improved_solution,
                "best_score": best_score,
                "metric_direction": metric_direction,
                "ablation_scripts": [],
                "functional_blocks": [],
                "ablation_results_list": [],
                "previous_summaries": ablation_summaries,
                "previous_blocks": [b for b in refined_blocks],
                "ablation_summaries": [],
                "target_block": "",
                "initial_plan": "",
                "status": "start",
            }

            with SubgraphSpan(
                "AblationSubgraph",
                input_data={
                    "outer_step": t,
                    "best_score": best_score,
                },
            ) as abl_span:
                ablation_result = ablation_subgraph.invoke(
                    ablation_input, abl_span.config
                )

                target_block = ablation_result.get("target_block", "")
                initial_plan = ablation_result.get("initial_plan", "")
                current_ablation_summaries = ablation_result.get(
                    "ablation_summaries", []
                )
                ablation_results_list = ablation_result.get("ablation_results_list", [])

                if current_ablation_summaries:
                    ablation_summaries.extend(current_ablation_summaries)
                if target_block and target_block not in refined_blocks:
                    refined_blocks.append(target_block)

                abl_span.set_output(
                    {
                        "target_block_name": target_block[:40],
                        "initial_plan_len": len(initial_plan),
                    }
                )

            if not target_block or not initial_plan:
                log_node_event(
                    "Algorithm2",
                    "cycle_skip",
                    {"outer_step": t, "reason": "no target block extracted"},
                )
                cycle_span.set_output({"outer_step": t, "status": "skipped_no_target"})
                continue

            # ── Inner loop: K refinement iterations ─────────────────────
            with SubgraphSpan(
                "RefinementSubgraph",
                input_data={
                    "outer_step": t,
                    "best_score": best_score,
                    "target_block": target_block[:40],
                },
            ) as ref_span:
                current_plan = initial_plan
                inner_step = 0
                debug_retries = 0
                current_improved_score = best_score
                current_improved_solution = improved_solution
                refined_code = ""

                for k in range(K):
                    with SubgraphSpan(
                        f"refinement_step_t{t}_k{k}",
                        input_data={
                            "inner_step": k,
                            "best_score": best_score,
                            "target_block": target_block[:40],
                        },
                    ) as step_span:
                        step_input: RefinementInnerState = {
                            "target_block": target_block,
                            "current_solution": current_improved_solution,
                            "best_score": best_score,
                            "metric_direction": metric_direction,
                            "initial_plan": initial_plan,
                            "current_plan": current_plan,
                            "refined_code": refined_code,
                            "candidate_solution": "",
                            "execution_output": "",
                            "execution_error": None,
                            "execution_score": None,
                            "inner_step": inner_step,
                            "debug_retries": debug_retries,
                            "improved_solution": current_improved_solution,
                            "improved_score": current_improved_score,
                            "status": "start" if k == 0 else "continue",
                        }

                        step_config = {
                            "configurable": {"thread_id": f"refine_t{t}_k{k}"}
                        }
                        step_result = refinement_step_subgraph.invoke(
                            step_input, step_config
                        )

                        step_status = step_result.get("status", "")
                        step_score = step_result.get("execution_score")
                        step_improved_score = step_result.get(
                            "improved_score", current_improved_score
                        )
                        step_improved_solution = step_result.get(
                            "improved_solution", current_improved_solution
                        )

                        if step_status == "error":
                            debug_retries = step_result.get(
                                "debug_retries", debug_retries + 1
                            )
                            step_span.set_output(
                                {
                                    "inner_step": k,
                                    "status": "error",
                                    "debug_retries": debug_retries,
                                }
                            )
                            if debug_retries >= MAX_DEBUG_RETRIES:
                                break
                            continue

                        if normalize_score(
                            step_improved_score, metric_direction
                        ) > normalize_score(current_improved_score, metric_direction):
                            current_improved_score = step_improved_score
                            current_improved_solution = step_improved_solution

                        current_plan = step_result.get("current_plan", current_plan)
                        inner_step = step_result.get("inner_step", k + 1)
                        debug_retries = step_result.get("debug_retries", 0)
                        refined_code = step_result.get("refined_code", "")

                        step_span.set_output(
                            {
                                "inner_step": k,
                                "status": step_status,
                                "improved_score": current_improved_score,
                            }
                        )

                ref_span.set_output(
                    {
                        "outer_step": t,
                        "refinement_steps": K,
                        "improved_score": current_improved_score,
                    }
                )

            if normalize_score(
                current_improved_score, metric_direction
            ) > normalize_score(best_score, metric_direction):
                best_score = current_improved_score
                improved_score = current_improved_score
                improved_solution = current_improved_solution

            cycle_span.set_output(
                {
                    "outer_step": t,
                    "improved_score": improved_score,
                    "best_score": best_score,
                }
            )

        log_node_event(
            "Algorithm2",
            "cycle_complete",
            {
                "outer_step": t,
                "improved_score": improved_score,
                "best_score": best_score,
            },
        )

    convergence_achieved = normalize_score(
        improved_score, metric_direction
    ) > normalize_score(state.get("best_score", 0), metric_direction)

    return {
        "improved_solution": improved_solution,
        "improved_score": improved_score,
        "convergence_achieved": convergence_achieved,
        "ablation_summaries": ablation_summaries,
        "refined_blocks": refined_blocks,
        "ablation_results_list": list(state.get("ablation_results_list", [])),
        "debug_history": debug_history,
        "security_violations": security_violations,
        "stage_history": stage_history,
        "outer_step": T,
        "status": "done",
    }


def run(
    initial_state: Dict = None,
    run_dir: str = None,
    thread_id: str = None,
    session_id: str = None,
):
    """Run Algorithm 2 with the given configuration.

    Wraps the entire pipeline execution in a root SubgraphSpan so all
    traces appear grouped under a single "Algorithm2" trace in Langfuse.

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
    phase_token = _current_phase.set("ablation")

    mle_logger = MLELogger(run_dir=run_dir)
    logger_token = _current_mle_logger.set(mle_logger)

    if initial_state is None:
        initial_state = {
            "current_solution": "# default mock solution",
            "best_score": 0.85,
            "metric_direction": "maximize",
            "ablation_scripts": [],
            "ablation_results_list": [],
            "ablation_summaries": [],
            "target_block": "",
            "initial_plan": "",
            "refined_blocks": [],
            "current_plans": [],
            "current_scores": [],
            "refined_code": "",
            "candidate_solution": "",
            "execution_output": "",
            "execution_error": None,
            "execution_score": None,
            "outer_step": 0,
            "inner_step": 0,
            "leakage_status": None,
            "leakage_code_block": None,
            "debug_history": [],
            "security_violations": [],
            "improved_solution": "",
            "improved_score": 0,
            "convergence_achieved": False,
            "stage_history": [],
            "status": "start",
        }

    if thread_id is None:
        thread_id = get_thread_id(run_dir)

    if session_id is None:
        session_id = thread_id

    log_node_event(
        "Algorithm2",
        "run_start",
        {"best_score": initial_state.get("best_score", 0), "run_dir": run_dir},
    )

    result: Dict = {}
    try:
        with SubgraphSpan(
            "Algorithm2",
            input_data={
                "best_score": initial_state.get("best_score", 0),
                "status": "start",
            },
        ) as root_span:
            with propagate_attributes(session_id=session_id):
                result = run_algorithm2(initial_state)
                root_span.set_output(
                    {
                        "status": result.get("status", ""),
                        "improved_score": result.get("improved_score", 0),
                        "outer_step": result.get("outer_step", 0),
                    }
                )
    finally:
        log_node_event(
            "Algorithm2",
            "run_end",
            {
                "status": result.get("status", ""),
                "improved_score": result.get("improved_score", 0),
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
                "Algorithm2", "flush_error", {"error": "langfuse.flush() failed"}
            )

    return result
