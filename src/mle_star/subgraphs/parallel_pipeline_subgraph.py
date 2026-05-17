"""Parallel pipeline fan-out subgraph (D15).

Dispatches L independent (Alg1+Alg2) pipeline runs via Send API fan-out,
producing L distinct candidate solutions for ensemble input.

Architecture:
    dispatch_parallel_pipelines(state) → L × Send("pipeline_flow", {...})
        → pipeline_flow_node(state) — runs full Alg1+Alg2 per pipeline
            → SubgraphSpan(f"pipeline_{i}") groups both algorithms:
                → SubgraphSpan(f"Algorithm_1") → Alg1 graph
                → SubgraphSpan(f"Algorithm_2") → Alg2 run_algorithm2
    Results accumulate via operator.add on parallel_results.

Desired Langfuse hierarchy:
    parallel_pipelines
    ├── pipeline_0
    │   ├── Algorithm_1
    │   │   └── {Algorithm 1 nodes}
    │   └── Algorithm_2
    │       └── {Algorithm 2 nodes}
    └── pipeline_1
        ├── Algorithm_1
        └── Algorithm_2

Pattern follows candidate_fanout_builder in algorithm_1.py.
"""

from typing import Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.mle_star.state.parallel_state import PipelineFlowState, ParallelFanoutState
from src.mle_star.state.shared import log_node_event, normalize_score, SubgraphSpan
from src.mle_star.config import NUM_PARALLEL_SOLUTIONS


def dispatch_parallel_pipelines(state: ParallelFanoutState) -> List[Send]:
    """Fan-out: dispatch one pipeline_flow per parallel solution (L pipelines).

    Creates L Send objects, one per independent (Alg1+Alg2) pipeline run.
    Each runs the full search + ablation+refinement process to produce a
    distinct candidate solution for ensemble input.
    """
    L = state.get("num_parallel_solutions", NUM_PARALLEL_SOLUTIONS)
    task_desc = state.get("task_desc", "")
    datasets = state.get("datasets", [])
    score_function_desc = state.get("score_function_desc", "")
    metric_direction = state.get("metric_direction", "maximize")

    sends = []
    for i in range(L):
        sends.append(
            Send(
                "pipeline_flow",
                {
                    "run_index": i,
                    "task_desc": task_desc,
                    "datasets": datasets,
                    "score_function_desc": score_function_desc,
                    "metric_direction": metric_direction,
                },
            )
        )
    return sends


def pipeline_flow_node(state: Dict) -> Dict:
    """Run full Alg1+Alg2 pipeline for one parallel solution.

    1. Opens a SubgraphSpan(f"pipeline_{i}") grouping span
    2. Invokes Alg1 graph inside SubgraphSpan("Algorithm_1")
    3. Maps Alg1 results via alg1_result_to_system
    4. Invokes Alg2 (run_algorithm2) inside SubgraphSpan("Algorithm_2")
    5. Maps Alg2 results via alg2_result_to_system
    6. Returns result dict with best_solution, best_score, etc.

    Each pipeline produces an independent candidate solution. Diversity
    comes from LLM stochasticity.
    """
    from src.mle_star.algorithms.algorithm_1 import get_graph
    from src.mle_star.algorithms.algorithm_2 import run_algorithm2
    from src.mle_star.graph import alg1_result_to_system, alg2_result_to_system
    from src.mle_star.state.shared import get_checkpointer, _current_run_dir

    i = state.get("run_index", 0)

    run_dir = _current_run_dir.get()
    alg1_checkpointer = get_checkpointer(run_dir) if run_dir else None

    with SubgraphSpan(
        f"pipeline_{i}",
        input_data={
            "pipeline": i,
            "task": state.get("task_desc", "")[:60],
        },
    ) as pipeline_span:
        # ── Phase 1: Algorithm 1 (Search) ────────────────────────────────
        alg1_state = {
            "task_desc": state.get("task_desc", "ML pipeline optimization"),
            "metric_direction": state.get("metric_direction", "maximize"),
            "retrieved_models": [],
            "candidates_pool": [],
            "leaderboard": [],
            "best_candidate": {},
            "current_reference_idx": 0,
            "stage_history": [],
            "status": "start",
        }
        alg1_config = {"configurable": {"thread_id": f"pipeline_{i}_alg1"}}

        with SubgraphSpan(
            "Algorithm_1",
            input_data={
                "task": alg1_state.get("task_desc", ""),
                "pipeline": i,
                "phase": "search",
            },
        ) as alg1_span:
            graph = get_graph(checkpointer=alg1_checkpointer)
            alg1_result = graph.invoke(alg1_state, alg1_config)
            alg1_span.set_output(
                {
                    "pipeline": i,
                    "status": alg1_result.get("status", ""),
                    "best_score": alg1_result.get("best_candidate", {}).get("score"),
                }
            )

        # ── Phase 2: Algorithm 2 (Ablation+Refinement) ───────────────────
        alg1_sys_updates = alg1_result_to_system(alg1_result)
        best_solution = alg1_sys_updates.get("best_solution", "")
        best_score = alg1_sys_updates.get("best_score", 0)

        alg2_state = {
            "current_solution": best_solution,
            "best_score": best_score,
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

        with SubgraphSpan(
            "Algorithm_2",
            input_data={
                "pipeline": i,
                "best_score": best_score,
                "phase": "ablation",
            },
        ) as alg2_span:
            alg2_result = run_algorithm2(alg2_state)
            alg2_span.set_output(
                {
                    "pipeline": i,
                    "status": alg2_result.get("status", ""),
                    "improved_score": alg2_result.get("improved_score", 0),
                }
            )

        # Use improved result if better (normalized), otherwise keep search result
        improved_solution = alg2_result.get("improved_solution", "") or best_solution
        improved_score = alg2_result.get("improved_score", 0)
        metric_direction = state.get("metric_direction", "maximize")
        if (
            normalize_score(improved_score, metric_direction)
            <= normalize_score(best_score, metric_direction)
            and improved_solution != best_solution
        ):
            improved_solution = best_solution
            improved_score = best_score

        pipeline_span.set_output(
            {
                "pipeline": i,
                "best_score": improved_score,
                "status": "done",
            }
        )

    return {
        "parallel_results": [
            {
                "run_index": i,
                "best_solution": improved_solution,
                "best_score": improved_score,
                "current_solution": improved_solution,
                "alg1_result": {
                    "status": alg1_result.get("status", ""),
                    "best_score": alg1_result.get("best_candidate", {}).get("score", 0),
                },
                "alg2_result": {
                    "status": alg2_result.get("status", ""),
                    "improved_score": alg2_result.get("improved_score", 0),
                },
            }
        ]
    }


# ── Build the compiled subgraph ──────────────────────────────────────────

_parallel_builder = StateGraph(ParallelFanoutState)

_parallel_builder.add_node("pipeline_flow", pipeline_flow_node)

_parallel_builder.add_conditional_edges(
    START,
    dispatch_parallel_pipelines,
    ["pipeline_flow"],
)
_parallel_builder.add_edge("pipeline_flow", END)

parallel_pipeline_subgraph = _parallel_builder.compile()


def get_parallel_pipeline_subgraph():
    """Build and return the compiled parallel pipeline subgraph.

    Topology:
        START → dispatch_parallel_pipelines (Send API fan-out)
                  ├── pipeline_flow (pipeline 0: Alg1 → Alg2)
                  ├── pipeline_flow (pipeline 1: Alg1 → Alg2)
                  └── ... (L pipelines total)
              → END (results accumulated via operator.add on parallel_results)
    """
    return parallel_pipeline_subgraph
