"""Main MLE-STAR system graph with supervisor and phase routing.

Builds the cyclic StateGraph that orchestrates parallel pipelines
(D15: L independent Alg1+Alg2 runs), ensemble, and submission phases.

Phase flow (D15): search → (parallel Alg1+Alg2 pipelines) → search_done
→ ensemble → submission → END

No backward restart to search. No stagnation edge (D12).
After max_full_cycles, force submission regardless.

Tracing architecture:
    - run() wraps the entire pipeline in a root SubgraphSpan named
      MLE_STAR_<timestamp> and a propagate_attributes(session_id=...) context,
      so ALL @traceable nodes and SubgraphSpans across ALL phases appear as
      nested children under ONE trace in ONE Langfuse session.
    - _current_obs ContextVar propagates through LangGraph's copy_context()
      so every node inside the graph sees the root span as parent.
"""

import logging
import os
from datetime import datetime
from typing import Dict

from langgraph.graph import END, START, StateGraph

from src.mle_star.state.system_state import MleStarSystemState
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
    infer_metric_direction,
    log_node_event,
    langfuse,
    normalize_score,
    propagate_attributes,
)
from src.mle_star.supervisor import SupervisorConfig, supervisor_decide, supervisor_node

_logger = logging.getLogger("mle_star")


# ── Phase Transition Functions ───────────────────────────────────────────


def transition_to_ensemble(state: MleStarSystemState | dict) -> dict:
    """Transition from search_done to ensemble phase.

    D15: Reads ensemble candidates from parallel_results (L independent
    pipeline runs). Falls back to [best_solution] if parallel_results
    is empty (backward compatibility / L=1 case).
    """
    parallel_results = state.get("parallel_results", [])

    if parallel_results:
        ensemble_solutions = [r.get("best_solution", "") for r in parallel_results]
        ensemble_input_scores = [r.get("best_score", 0) for r in parallel_results]
    else:
        best_solution = state.get("best_solution") or ""
        ensemble_solutions = [best_solution]
        ensemble_input_scores = [state.get("best_score", 0)]

    _current_phase.set("ensemble")

    log_node_event(
        "transition_to_ensemble",
        "transition",
        {
            "from_phase": state.get("phase"),
            "to_phase": "ensemble",
            "best_score": state.get("best_score"),
            "num_solutions": len(ensemble_solutions),
        },
    )

    return {
        "phase": "ensemble",
        "ensemble_solutions": ensemble_solutions,
        "ensemble_input_scores": ensemble_input_scores,
        "ensemble_round": 0,
        "full_cycles": state.get("full_cycles", 0) + 1,
    }


def transition_to_submission(state: MleStarSystemState | dict) -> dict:
    """Transition to submission phase."""
    best_solution = state.get("best_solution") or ""

    _current_phase.set("submission")

    log_node_event(
        "transition_to_submission",
        "transition",
        {
            "from_phase": state.get("phase"),
            "to_phase": "submission",
            "best_score": state.get("best_score"),
        },
    )

    return {
        "phase": "submission",
        "submission_code": best_solution,
    }


# ── State Mapping Functions ──────────────────────────────────────────────


def system_to_alg1_state(state: MleStarSystemState | dict) -> dict:
    """Map MleStarSystemState to Alg1State for Algorithm 1 invocation."""
    return {
        "task_desc": state.get("task_desc", ""),
        "metric_direction": state.get("metric_direction", "maximize"),
        "retrieved_models": state.get("alg1_result", {}).get("retrieved_models", []),
        "candidates_pool": [],
        "leaderboard": [],
        "best_candidate": {},
        "current_reference_idx": 0,
        "stage_history": [],
        "status": "start",
    }


def alg1_result_to_system(result: dict) -> dict:
    """Map Algorithm 1 output back to MleStarSystemState updates."""
    best_candidate = result.get("best_candidate", {})
    best_score = best_candidate.get("score", 0)
    best_code = best_candidate.get("code", "")

    return {
        "alg1_result": result,
        "best_solution": best_code,
        "best_score": best_score,
        "raw_best_score": best_score,
        "current_solution": best_code,
        "current_score": best_score,
        "phase": "search_done",
        "status": "search_complete",
    }


def system_to_alg2_state(state: MleStarSystemState | dict) -> dict:
    """Map MleStarSystemState to Alg2State for Algorithm 2 invocation.

    Resets operator.add-accumulated fields to empty lists/collections
    to prevent double-accumulation when the outer-loop Python code
    extends these lists manually.
    """
    return {
        "current_solution": state.get("current_solution", ""),
        "best_score": state.get("best_score", 0),
        "metric_direction": state.get("metric_direction", "maximize"),
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
        "outer_step": state.get("outer_step", 0),
        "inner_step": state.get("inner_step", 0),
        "leakage_status": None,
        "leakage_code_block": None,
        "debug_history": [],
        "security_violations": [],
        "debug_retries": 0,
        "best_ensemble_code": "",
        "best_ensemble_score": 0,
        "improved_solution": "",
        "improved_score": 0,
        "convergence_achieved": False,
        "stage_history": [],
        "status": "start",
    }


def alg2_result_to_system(result: dict, state: MleStarSystemState | dict) -> dict:
    """Map Algorithm 2 output back to MleStarSystemState updates."""
    improved_solution = result.get("improved_solution", "")
    improved_score = result.get("improved_score", 0)
    current_best_score = state.get("best_score", 0)
    metric_direction = state.get("metric_direction", "maximize")

    updates = {
        "alg2_result": result,
        "current_solution": improved_solution or state.get("current_solution", ""),
        "current_score": improved_score,
        "stage_history": result.get("stage_history", []),
    }

    if (
        normalize_score(improved_score, metric_direction)
        > normalize_score(current_best_score, metric_direction)
        and improved_solution
    ):
        updates["best_solution"] = improved_solution
        updates["best_score"] = improved_score
        updates["raw_best_score"] = improved_score

    updates["outer_step"] = result.get("outer_step", state.get("outer_step", 0))
    updates["inner_step"] = result.get("inner_step", 0)
    updates["convergence_achieved"] = result.get("convergence_achieved", False)

    if result.get("security_violations"):
        updates["security_violations"] = result.get("security_violations", [])

    return updates


def system_to_alg3_state(state: MleStarSystemState | dict) -> dict:
    """Map MleStarSystemState to Alg3State for Algorithm 3 invocation."""
    parallel_results = state.get("parallel_results", [])
    if parallel_results:
        ensemble_input_scores = [r.get("best_score", 0) for r in parallel_results]
    else:
        ensemble_input_scores = [state.get("best_score", 0)]

    return {
        "ensemble_solutions": state.get("ensemble_solutions", []),
        "ensemble_input_scores": ensemble_input_scores,
        "metric_direction": state.get("metric_direction", "maximize"),
        "ensemble_plans": [],
        "ensemble_scores": [],
        "current_ensemble_plan": "",
        "current_ensemble_code": "",
        "current_ensemble_score": 0,
        "ensemble_round": state.get("ensemble_round", 0),
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


def alg3_result_to_system(result: dict, state: MleStarSystemState | dict) -> dict:
    """Map Algorithm 3 output back to MleStarSystemState updates."""
    best_ensemble_code = result.get("best_ensemble_code", "")
    best_ensemble_score = result.get("best_ensemble_score", 0)
    current_best_score = state.get("best_score", 0)
    metric_direction = state.get("metric_direction", "maximize")

    updates = {
        "alg3_result": result,
        "ensemble_round": result.get("ensemble_round", state.get("ensemble_round", 0))
        + 1,
        "stage_history": result.get("stage_history", []),
    }

    if (
        normalize_score(best_ensemble_score, metric_direction)
        > normalize_score(current_best_score, metric_direction)
        and best_ensemble_code
    ):
        updates["best_solution"] = best_ensemble_code
        updates["best_score"] = best_ensemble_score
        updates["raw_best_score"] = best_ensemble_score

    return updates


# ── Subgraph Node Wrappers ───────────────────────────────────────────────


def alg1_subgraph_node(state: MleStarSystemState) -> dict:
    """Invoke Algorithm 1 subgraph and map results back to system state.

    Wraps the invocation in SubgraphSpan("Alg1__search") so all
    @traceable nodes inside Algorithm 1 appear grouped under one
    parent span in Langfuse.
    """
    from src.mle_star.algorithms.algorithm_1 import get_graph
    from src.mle_star.state.shared import get_checkpointer, _current_run_dir

    alg1_state = system_to_alg1_state(state)

    run_dir = _current_run_dir.get()
    alg1_checkpointer = get_checkpointer(run_dir) if run_dir else None

    graph = get_graph(checkpointer=alg1_checkpointer)
    config = {"configurable": {"thread_id": f"alg1_{state.get('status', 'run')}"}}
    with SubgraphSpan(
        "Alg1__search",
        input_data={"task": alg1_state.get("task_desc", ""), "phase": "search"},
    ) as span:
        result = graph.invoke(alg1_state, config)
        span.set_output(
            {
                "status": result.get("status", ""),
                "best_score": result.get("best_candidate", {}).get("score"),
            }
        )
    return alg1_result_to_system(result)


def alg2_subgraph_node(state: MleStarSystemState) -> dict:
    """Invoke Algorithm 2 subgraph and map results back to system state.

    Wraps the invocation in SubgraphSpan("Alg2__refine") so all
    @traceable nodes inside Algorithm 2 appear grouped under one
    parent span in Langfuse.

    Algorithm 2 runs the ablation+refinement outer loop internally
    via a Python-level for loop with SubgraphSpan per iteration.
    """
    alg2_state = system_to_alg2_state(state)
    from src.mle_star.algorithms.algorithm_2 import run_algorithm2

    with SubgraphSpan(
        "Alg2__refine",
        input_data={
            "solution": alg2_state.get("current_solution", ""),
            "phase": "ablation",
        },
    ) as span:
        result = run_algorithm2(alg2_state)
        span.set_output(
            {
                "status": result.get("status", ""),
                "improved_score": result.get("improved_score", 0),
                "outer_step": result.get("outer_step", 0),
            }
        )
    updates = alg2_result_to_system(result, state)

    return updates


def parallel_pipeline_node(state: MleStarSystemState) -> dict:
    """Invoke L parallel (Alg1+Alg2) pipelines via Send API fan-out.

    D15: Dispatches independent pipeline runs, aggregates results,
    picks global best, transitions to search_done phase.

    Wraps the fan-out in SubgraphSpan("parallel_pipelines") so all
    pipeline_flow traces appear grouped under one parent span.
    """
    from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
        get_parallel_pipeline_subgraph,
    )

    num_parallel = state.get("num_parallel_solutions", None)
    if num_parallel is None:
        from src.mle_star.config import NUM_PARALLEL_SOLUTIONS

        num_parallel = NUM_PARALLEL_SOLUTIONS

    fanout_state = {
        "task_desc": state.get("task_desc", ""),
        "datasets": state.get("datasets", []),
        "score_function_desc": state.get("score_function_desc", ""),
        "metric_direction": state.get("metric_direction", "maximize"),
        "num_parallel_solutions": num_parallel,
        "parallel_results": [],
    }

    log_node_event(
        "parallel_pipeline_node",
        "dispatch",
        {"num_parallel_solutions": num_parallel, "phase": state.get("phase")},
    )

    with SubgraphSpan(
        "parallel_pipelines",
        input_data={
            "num_parallel_solutions": num_parallel,
            "task": state.get("task_desc", "")[:60],
            "phase": "search",
        },
    ) as span:
        subgraph = get_parallel_pipeline_subgraph()
        config = {
            "configurable": {"thread_id": f"parallel_{state.get('status', 'run')}"}
        }
        result = subgraph.invoke(fanout_state, config)
        span.set_output(
            {
                "num_results": len(result.get("parallel_results", [])),
                "phase": "search_done",
            }
        )

    parallel_results = result.get("parallel_results", [])

    global_best_solution = state.get("best_solution", "")
    global_best_score = state.get("best_score", 0)
    metric_direction = state.get("metric_direction", "maximize")

    for r in parallel_results:
        score = r.get("best_score", 0)
        solution = r.get("best_solution", "")
        if (
            normalize_score(score, metric_direction)
            > normalize_score(global_best_score, metric_direction)
            and solution
        ):
            global_best_score = score
            global_best_solution = solution

    _current_phase.set("search_done")

    log_node_event(
        "parallel_pipeline_node",
        "complete",
        {
            "num_results": len(parallel_results),
            "global_best_score": global_best_score,
            "phase": "search_done",
        },
    )

    all_stage_history = []
    for r in parallel_results:
        for key in ("alg1_result", "alg2_result"):
            sub_result = r.get(key, {})
            if isinstance(sub_result, dict) and "stage_history" in sub_result:
                all_stage_history.extend(sub_result["stage_history"])

    updates = {
        "parallel_results": parallel_results,
        "best_solution": global_best_solution,
        "best_score": global_best_score,
        "raw_best_score": global_best_score,
        "current_solution": global_best_solution,
        "current_score": global_best_score,
        "phase": "search_done",
        "status": "search_complete",
        "stage_history": all_stage_history,
    }

    if parallel_results:
        updates["alg1_result"] = parallel_results[0].get("alg1_result", {})

    return updates


def alg3_subgraph_node(state: MleStarSystemState) -> dict:
    """Invoke Algorithm 3 and map results back to system state.

    Algorithm 3 manages its own round loop internally via run_algorithm3(),
    matching the pattern used by Algorithm 2. All R ensemble rounds appear
    as nested children under one Alg3__ensemble SubgraphSpan in Langfuse.
    """
    alg3_state = system_to_alg3_state(state)
    from src.mle_star.algorithms.algorithm_3 import run_algorithm3

    with SubgraphSpan(
        "Alg3__ensemble",
        input_data={
            "solutions": len(alg3_state.get("ensemble_solutions", [])),
            "phase": "ensemble",
        },
    ) as span:
        result = run_algorithm3(alg3_state)
        span.set_output(
            {
                "status": result.get("status", ""),
                "best_ensemble_score": result.get("best_ensemble_score", 0),
                "ensemble_round": result.get("ensemble_round", 0),
            }
        )
    updates = alg3_result_to_system(result, state)

    return updates


def submission_node(state: MleStarSystemState) -> dict:
    """Invoke submission subgraph and map results back to system state.

    Wraps the invocation in SubgraphSpan("submission") so the
    submission node appears grouped in Langfuse.

    State mapping per Section 8.6: passes final_solution (best_solution),
    task_desc, and score_function_desc for A_test prompt context.
    """
    from src.mle_star.subgraphs.submission_subgraph import get_submission_subgraph

    sub_state = {
        "final_solution": state.get("best_solution", ""),
        "best_score": state.get("best_score", 0),
        "task_desc": state.get("task_desc", ""),
        "score_function_desc": state.get("score_function_desc", ""),
        "submission_code": "",
        "submission_score": None,
        "subsampling_block": "",
        "leakage_status": None,
        "final_dir": "",
        "status": "start",
    }
    submission_interrupts = state.get("_submission_interrupt_before", None)
    graph = get_submission_subgraph(interrupt_before=submission_interrupts)
    with SubgraphSpan(
        "submission",
        input_data={
            "best_solution": sub_state["final_solution"][:60],
            "phase": "submission",
        },
    ) as span:
        result = graph.invoke(sub_state)
        span.set_output(
            {
                "submission_code": result.get("submission_code", "")[:60],
                "submission_score": result.get("submission_score"),
            }
        )
    return {
        "submission_code": result.get("submission_code", ""),
        "submission_score": result.get("submission_score"),
        "final_dir": result.get("final_dir", ""),
        "status": "done",
    }


# ── Graph Construction ──────────────────────────────────────────────────


SUPERVISOR_ROUTING_MAP = {
    "parallel_pipeline_node": "parallel_pipeline_node",
    "continue_ablation": "alg2_subgraph",
    "new_ablation_cycle": "alg2_subgraph",
    "transition_to_ensemble": "transition_to_ensemble",
    "transition_to_submission": "transition_to_submission",
    "END": END,
}


def get_mle_star_graph(
    config: SupervisorConfig | None = None,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):
    """Build and return the compiled MLE-STAR main graph.

    Topology (D15):
        START → supervisor
                  ├─(parallel_pipeline_node)──► parallel_pipeline_node ──► supervisor
                  ├─(transition_to_ensemble)──► transition_to_ensemble ──► alg3_subgraph ──► supervisor
                  ├─(continue_ablation)────────► alg2_subgraph ──► supervisor
                  ├─(new_ablation_cycle)──────► alg2_subgraph ──► supervisor
                  ├─(transition_to_submission)► transition_to_submission ──► END

    Note: alg1_subgraph and alg2_subgraph are invoked inside pipeline_flow_node
    (parallel_pipeline_subgraph), not as main graph nodes.
    Algorithm 3 manages its own round loop internally (R rounds via Python for loop).
    The supervisor does NOT loop back to alg3_subgraph — once Alg3 runs, it
    transitions to submission.

    Human-in-the-loop (Stage 10):
        interrupt_before: List of node names to pause before execution.
        When set (with a checkpointer), graph.invoke() raises a GraphInterrupt
        before executing the named node. The caller can inspect/modify state
        and resume by calling graph.invoke(None, config) again.

    Args:
        config: Optional SupervisorConfig for routing thresholds.
        checkpointer: Optional checkpointer for state persistence.
            When provided, state is saved after every superstep, enabling
            resume from any phase. Pass None (default) to compile without
            checkpointing.
        interrupt_before: Optional list of node names to interrupt before.
            If None and config has interrupt_points, uses those instead.
    """
    if config is None:
        config = SupervisorConfig()

    builder = StateGraph(MleStarSystemState)

    def _supervisor_node_with_config(state: MleStarSystemState) -> dict:
        return supervisor_node(state, config)

    builder.add_node("supervisor", _supervisor_node_with_config)
    builder.add_node("parallel_pipeline_node", parallel_pipeline_node)
    builder.add_node("alg2_subgraph", alg2_subgraph_node)
    builder.add_node("transition_to_ensemble", transition_to_ensemble)
    builder.add_node("alg3_subgraph", alg3_subgraph_node)
    builder.add_node("transition_to_submission", transition_to_submission)
    builder.add_node("submission_node", submission_node)

    builder.add_edge(START, "supervisor")

    def route_supervisor(state: MleStarSystemState) -> str:
        decision = supervisor_decide(state, config)
        return decision

    builder.add_conditional_edges(
        "supervisor", route_supervisor, SUPERVISOR_ROUTING_MAP
    )

    builder.add_edge("parallel_pipeline_node", "supervisor")
    builder.add_edge("alg2_subgraph", "supervisor")
    builder.add_edge("transition_to_ensemble", "alg3_subgraph")
    builder.add_edge("alg3_subgraph", "supervisor")
    builder.add_edge("transition_to_submission", "submission_node")
    builder.add_edge("submission_node", END)

    compile_kwargs = {"name": "MLE_STAR"}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    effective_interrupt = (
        interrupt_before or getattr(config, "interrupt_points", None) or []
    )
    if effective_interrupt:
        compile_kwargs["interrupt_before"] = effective_interrupt

    return builder.compile(**compile_kwargs)


def get_alg1_graph(checkpointer=None):
    """Lazily import and return the Algorithm 1 compiled graph.

    This avoids module-level side effects (checkpoint directory creation)
    at import time.
    """
    from src.mle_star.algorithms.algorithm_1 import get_graph

    return get_graph(checkpointer=checkpointer)


def run(
    initial_state: Dict | None = None,
    run_dir: str | None = None,
    thread_id: str | None = None,
    session_id: str | None = None,
    config: SupervisorConfig | None = None,
    resume: bool = False,
) -> Dict:
    """Run the full MLE-STAR pipeline with tracing, session grouping, and
    checkpointing.

    Wraps the entire pipeline invocation in a root SubgraphSpan named
    MLE_STAR_<timestamp> and a propagate_attributes(session_id=...) context,
    so all observations across all phases appear under one trace in one
    Langfuse session.

    Creates a timestamped run directory and initializes a per-run MLELogger
    that writes JSON events to runs/{run_dir}/run.log.

    Checkpointing:
        A checkpointer is always created via get_checkpointer(run_dir). State
        is persisted after every superstep (node execution). The thread_id in
        the config identifies the checkpoint thread. When resume=True, the
        graph is invoked with None to continue from the last checkpoint rather
        than starting fresh.

    Args:
        initial_state: Optional initial state dict. Defaults to a minimal
            state with task_desc and search phase. Ignored when resume=True.
        run_dir: Optional run directory path. Auto-generated if not provided.
            Must match a previous run_dir when resume=True.
        thread_id: Optional thread ID for checkpointing. Defaults to the
            run_dir basename. Must match a previous thread_id when resume=True.
        session_id: Optional session ID for Langfuse session grouping.
            Defaults to thread_id.
        config: Optional SupervisorConfig for routing thresholds.
        resume: If True, resume from the last checkpoint for the given
            thread_id instead of starting fresh. initial_state is ignored.

    Returns:
        Final state dict after pipeline execution completes.
    """
    if run_dir is None:
        run_dir = generate_run_dir()

    os.makedirs(run_dir, exist_ok=True)
    run_id = os.path.basename(run_dir.rstrip("/"))

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    trace_name = f"MLE_STAR_{timestamp}"

    run_dir_token = _current_run_dir.set(run_dir)
    run_id_token = _current_run_id.set(run_id)
    phase_token = _current_phase.set("search")

    mle_logger = MLELogger(run_dir=run_dir)
    logger_token = _current_mle_logger.set(mle_logger)

    checkpointer = get_checkpointer(run_dir)

    if config is None:
        config = SupervisorConfig()

    if not resume and initial_state is None:
        initial_state = {
            "task_desc": "ML pipeline optimization",
            "datasets": [],
            "score_function_desc": "",
            "phase": "search",
            "phase_history": [],
            "current_solution": "",
            "best_solution": "",
            "best_score": 0.0,
            "current_score": None,
            "metric_direction": "maximize",
            "raw_best_score": None,
            "alg1_result": {},
            "parallel_results": [],
            "alg2_result": {},
            "outer_step": 0,
            "inner_step": 0,
            "convergence_achieved": False,
            "alg3_result": {},
            "ensemble_solutions": [],
            "ensemble_input_scores": [],
            "ensemble_round": 0,
            "submission_code": "",
            "submission_score": None,
            "final_dir": "",
            "full_cycles": 0,
            "max_full_cycles": config.max_full_cycles,
            "debug_history": [],
            "security_violations": [],
            "stage_history": [],
            "status": "start",
        }

    if not resume and initial_state is not None:
        score_desc = initial_state.get("score_function_desc", "")
        if not initial_state.get("metric_direction"):
            initial_state["metric_direction"] = infer_metric_direction(score_desc)

    if thread_id is None:
        thread_id = get_thread_id(run_dir)

    if session_id is None:
        session_id = thread_id

    log_node_event(
        "MLE_STAR",
        "run_start" if not resume else "run_resume",
        {
            "task": (initial_state or {}).get("task_desc", ""),
            "run_dir": run_dir,
            "resume": resume,
            "thread_id": thread_id,
        },
    )

    thread_config = {"configurable": {"thread_id": thread_id}}
    graph = get_mle_star_graph(config, checkpointer=checkpointer)

    result: Dict = {}
    try:
        with SubgraphSpan(
            trace_name,
            input_data={
                "task": (initial_state or {}).get("task_desc", ""),
                "status": "resuming" if resume else "start",
                "phase": (initial_state or {}).get("phase", "search"),
            },
        ) as root_span:
            with propagate_attributes(session_id=session_id):
                if resume:
                    result = graph.invoke(None, thread_config)
                else:
                    result = graph.invoke(initial_state, thread_config)
                root_span.set_output(
                    {
                        "status": result.get("status", ""),
                        "best_score": result.get("best_score"),
                        "phase": result.get("phase", ""),
                        "submission_code": result.get("submission_code", "")[:60],
                    }
                )
    finally:
        result["__checkpointer__"] = checkpointer
        result["__thread_config__"] = thread_config
        log_node_event(
            "MLE_STAR",
            "run_end",
            {
                "status": result.get("status", ""),
                "phase": result.get("phase", ""),
                "best_score": result.get("best_score"),
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
                "MLE_STAR", "flush_error", {"error": "langfuse.flush() failed"}
            )

    return result


def resume_with_update(
    graph, thread_config: dict, state_updates: dict | None = None
) -> Dict:
    """Resume a paused graph execution, optionally applying state updates first.

    Used for human-in-the-loop: when the graph is paused at an interrupt
    point, the caller can modify state and then resume execution.

    Args:
        graph: The compiled MLE-STAR graph.
        thread_config: Thread config dict with thread_id for checkpointing.
        state_updates: Optional dict of state fields to update before
            resuming. Applied via graph.update_state().

    Returns:
        Final state dict after graph execution completes.
    """
    if state_updates:
        graph.update_state(thread_config, state_updates)

    log_node_event(
        "resume_with_update",
        "resuming",
        {
            "thread_id": thread_config.get("configurable", {}).get("thread_id", ""),
            "updates": list(state_updates.keys()) if state_updates else [],
        },
    )

    result = graph.invoke(None, thread_config)

    return result
