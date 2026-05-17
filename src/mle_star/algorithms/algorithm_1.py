"""Parent graph for Algorithm 1 (Initial Solution Generation).

Orchestrates the full Algorithm 1 flow:

    A1 Retrieve models
        → generate candidates (compiled fan-out subgraph with Send API)
            → candidates_pool reduced via operator.add
        → rank
        → merge candidates (compiled fan-out subgraph with Send API)
            → candidates_pool reduced via operator.add
        → select_best

The candidate and merge fan-out subgraphs use LangGraph's Send API for
parallel execution. They are compiled StateGraphs added as nodes to the
parent Algorithm1 graph.

Tracing architecture:
    - run() wraps the entire pipeline in a root SubgraphSpan ("Algorithm1")
    - generate_candidates_node is @traceable and creates a SubgraphSpan
      ("candidates_group") that encompasses all fan-out tasks
    - merge_candidates_node is @traceable and creates a SubgraphSpan
      ("merges_group") that encompasses all merge fan-out tasks
    - candidate_flow_node and merge_flow_node use SubgraphSpan directly
      (no redundant @traceable decorator; the SubgraphSpan provides grouping)
    - _current_obs context var propagates through LangGraph's copy_context()
      in ThreadPoolExecutor, so fan-out tasks see the grouping span as parent
    - propagate_attributes(session_id=...) sets the session ID in the OTel
      context, so all observations are grouped in the Langfuse Sessions view
    - Expected Langfuse hierarchy:
        Algorithm1
        ├── A1__retrieve
        ├── generate_candidates
        │   └── candidates_group
        │       ├── candidate_subgraph (LGBM)
        │       ├── candidate_subgraph (XGB)
        │       └── candidate_subgraph (Torch)
        ├── Rank
        ├── merge_candidates
        │   └── merges_group
        │       ├── merge_subgraph (pair 1)
        │       └── merge_subgraph (pair 2)
        └── SelectBest
"""

import os
from typing import Annotated, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.mle_star.state.alg1_state import Alg1State, FanoutState, MergeFanoutState
from src.mle_star.state.shared import (
    MLELogger,
    generate_run_dir,
    get_checkpointer,
    get_run_dir,
    get_thread_id,
    log_node_event,
    normalize_score,
    propagate_attributes,
    simulate_delay,
    traceable,
    SubgraphSpan,
    langfuse,
    _current_run_id,
    _current_run_dir,
    _current_phase,
    _current_mle_logger,
    reset_llm_call_counter,
)
from src.mle_star.nodes.search import A1__retrieve as _real_A1__retrieve
from src.mle_star.subgraphs.candidate_subgraph import candidate_subgraph
from src.mle_star.subgraphs.merge_subgraph import merge_subgraph


@traceable("A1__retrieve")
def A1__retrieve(state: Alg1State) -> Dict:
    """Retrieve model candidates for evaluation.

    Delegates to the real implementation in nodes/search.py which
    uses SearchCache + LLM. Mock mode is handled inside that function.
    """
    return _real_A1__retrieve(state)


@traceable("Rank")
def rank_node(state: Alg1State) -> Dict:
    """Rank candidates by score and identify the best candidate.

    Sorts candidates_pool by normalized score (higher is better after
    normalization) and sets the best_candidate. Updates leaderboard
    for merge reference.
    """
    simulate_delay()
    pool = state.get("candidates_pool", [])
    if not pool:
        return {"leaderboard": [], "best_candidate": {}}

    metric_direction = state.get("metric_direction", "maximize")
    sorted_pool = sorted(
        pool,
        key=lambda x: normalize_score(x.get("score", 0), metric_direction),
        reverse=True,
    )
    best = sorted_pool[0]
    return {"leaderboard": sorted_pool, "best_candidate": best}


@traceable("SelectBest")
def select_best_node(state: Alg1State) -> Dict:
    """Select the final best candidate from the pool.

    Finds the candidate with the highest normalized score and returns
    it as best_candidate with status='done'.
    """
    simulate_delay()
    pool = state.get("candidates_pool", [])
    if not pool:
        return {"status": "done"}

    metric_direction = state.get("metric_direction", "maximize")
    best = max(pool, key=lambda x: normalize_score(x.get("score", 0), metric_direction))
    return {"best_candidate": best, "status": "done"}


def dispatch_candidates(state: FanoutState) -> List[Send]:
    """Fan-out: dispatch one candidate subgraph per retrieved model."""
    model_descriptions = state.get("model_descriptions", [])
    model_desc_map = {
        m.get("model_name", ""): m for m in model_descriptions if isinstance(m, dict)
    }
    sends = [
        Send(
            "candidate_flow",
            {
                "model": model,
                "model_description": model_desc_map.get(model, {}),
                "task_desc": state.get("task_desc", ""),
                "score_function_desc": state.get("score_function_desc", ""),
                "datasets": state.get("datasets", []),
                "metric_direction": state.get("metric_direction", "maximize"),
            },
        )
        for model in state.get("retrieved_models", [])
    ]
    return sends


def candidate_flow_node(state: Dict) -> Dict:
    """Invoke candidate subgraph for one model.

    Runs the full candidate generation pipeline (A2→A13→A12→A11)
    for a single model. Wrapped in a SubgraphSpan for grouped tracing.
    """
    model = state["model"]
    model_description = state.get("model_description") or {}
    task_desc = state.get("task_desc", "")
    score_function_desc = state.get("score_function_desc", "")
    datasets = state.get("datasets", [])
    metric_direction = state.get("metric_direction", "maximize")

    with SubgraphSpan(
        "candidate_subgraph",
        model=model,
        input_data={"model": model, "status": "pending"},
    ) as span:
        sub_state = {
            "model": model,
            "model_description": model_description,
            "task_desc": task_desc,
            "score_function_desc": score_function_desc,
            "datasets": datasets,
            "metric_direction": metric_direction,
            "attempts": 0,
            "usage_fix_attempts": 0,
            "leakage_fix_attempts": 0,
            "execution_output": "",
            "execution_error": None,
            "sub_events": [],
            "status": "pending",
            "code": "",
            "score": 0.0,
        }
        result = candidate_subgraph.invoke(sub_state, span.config)
        span.set_output(
            {
                "model": model,
                "code": result.get("code", "")[:80],
                "score": result.get("score", 0.0),
            }
        )

    return {
        "candidates_pool": [
            {
                "model": model,
                "code": result.get("code", ""),
                "score": result.get("score", 0.0),
            }
        ]
    }


candidate_fanout_builder = StateGraph(FanoutState)
candidate_fanout_builder.add_node("candidate_flow", candidate_flow_node)
candidate_fanout_builder.add_conditional_edges(
    START, dispatch_candidates, ["candidate_flow"]
)
candidate_fanout_builder.add_edge("candidate_flow", END)
candidate_fanout_subgraph = candidate_fanout_builder.compile()


def dispatch_merges(state: MergeFanoutState) -> List[Send]:
    """Fan-out: dispatch one merge subgraph per (best, reference) pair."""
    best = state.get("best_candidate", {})
    leaderboard = state.get("leaderboard", [])
    refs = [c for c in leaderboard if c.get("code") != best.get("code")]
    task_desc = state.get("task_desc", "")
    score_function_desc = state.get("score_function_desc", "")
    datasets = state.get("datasets", [])
    metric_direction = state.get("metric_direction", "maximize")

    if not best or not refs:
        return []

    sends = [
        Send(
            "merge_flow",
            {
                "base_code": best.get("code", ""),
                "ref_code": ref.get("code", ""),
                "task_desc": task_desc,
                "score_function_desc": score_function_desc,
                "datasets": datasets,
                "metric_direction": metric_direction,
            },
        )
        for ref in refs
    ]
    return sends


def merge_flow_node(state: Dict) -> Dict:
    """Invoke merge subgraph for one (best, reference) pair.

    Runs the merge pipeline (A3→A12→A11) for a single pair.
    Wrapped in a SubgraphSpan for grouped tracing.
    """
    base_code = state["base_code"]
    ref_code = state["ref_code"]
    task_desc = state.get("task_desc", "")
    score_function_desc = state.get("score_function_desc", "")
    datasets = state.get("datasets", [])
    metric_direction = state.get("metric_direction", "maximize")

    with SubgraphSpan(
        "merge_subgraph",
        input_data={
            "base_code": base_code[:60],
            "ref_code": ref_code[:60],
        },
    ) as span:
        sub_state = {
            "base_code": base_code,
            "ref_code": ref_code,
            "merged_code": "",
            "task_desc": task_desc,
            "score_function_desc": score_function_desc,
            "datasets": datasets,
            "metric_direction": metric_direction,
            "score": 0.0,
            "attempts": 0,
            "leakage_fix_attempts": 0,
            "execution_output": "",
            "execution_error": None,
            "sub_events": [],
            "status": "pending",
        }
        result = merge_subgraph.invoke(sub_state, span.config)
        span.set_output(
            {
                "merged_code": result.get("merged_code", "")[:80],
                "score": result.get("score", 0.0),
            }
        )

    return {
        "candidates_pool": [
            {
                "model": "merged",
                "code": result.get("merged_code", ""),
                "score": result.get("score", 0.0),
            }
        ]
    }


merge_fanout_builder = StateGraph(MergeFanoutState)
merge_fanout_builder.add_node("merge_flow", merge_flow_node)
merge_fanout_builder.add_conditional_edges(START, dispatch_merges, ["merge_flow"])
merge_fanout_builder.add_edge("merge_flow", END)
merge_fanout_subgraph = merge_fanout_builder.compile()


@traceable("generate_candidates")
def generate_candidates_node(state: Alg1State) -> Dict:
    """Generate candidate solutions for all retrieved models.

    Invokes the candidate fan-out subgraph (Send API) inside a grouping
    SubgraphSpan so all candidate_flow traces appear nested under
    this node in Langfuse.
    """
    with SubgraphSpan(
        "candidates_group",
        input_data={"models": state.get("retrieved_models", [])},
    ) as group_span:
        fanout_state = {
            "retrieved_models": state.get("retrieved_models", []),
            "model_descriptions": state.get("model_descriptions", []),
            "task_desc": state.get("task_desc", ""),
            "score_function_desc": state.get("score_function_desc", ""),
            "datasets": state.get("datasets", []),
            "metric_direction": state.get("metric_direction", "maximize"),
            "candidates_pool": [],
        }
        fanout_config = {
            "configurable": {"thread_id": f"cand_{group_span.trace_id[:8]}"}
        }
        result = candidate_fanout_subgraph.invoke(fanout_state, fanout_config)
        group_span.set_output(
            {
                "candidates_count": len(result.get("candidates_pool", [])),
                "models": state.get("retrieved_models", []),
            }
        )

    return {"candidates_pool": result.get("candidates_pool", [])}


@traceable("merge_candidates")
def merge_candidates_node(state: Alg1State) -> Dict:
    """Merge best candidate with each reference candidate.

    Invokes the merge fan-out subgraph (Send API) inside a grouping
    SubgraphSpan so all merge_flow traces appear nested under this node
    in Langfuse.
    """
    best = state.get("best_candidate", {})
    leaderboard = state.get("leaderboard", [])
    refs = [c for c in leaderboard if c.get("code") != best.get("code")]

    if not best or not refs:
        return {"candidates_pool": []}

    with SubgraphSpan(
        "merges_group",
        input_data={
            "best_code": best.get("code", "")[:60],
            "ref_count": len(refs),
        },
    ) as group_span:
        fanout_state = {
            "best_candidate": best,
            "leaderboard": leaderboard,
            "task_desc": state.get("task_desc", ""),
            "score_function_desc": state.get("score_function_desc", ""),
            "datasets": state.get("datasets", []),
            "metric_direction": state.get("metric_direction", "maximize"),
            "candidates_pool": [],
        }
        fanout_config = {
            "configurable": {"thread_id": f"merge_{group_span.trace_id[:8]}"}
        }
        result = merge_fanout_subgraph.invoke(fanout_state, fanout_config)
        group_span.set_output(
            {
                "merge_count": len(result.get("candidates_pool", [])),
            }
        )

    return {"candidates_pool": result.get("candidates_pool", [])}


builder = StateGraph(Alg1State)

builder.add_node("A1__retrieve", A1__retrieve)
builder.add_node("generate_candidates", generate_candidates_node)
builder.add_node("Rank", rank_node)
builder.add_node("merge_candidates", merge_candidates_node)
builder.add_node("SelectBest", select_best_node)

builder.add_edge(START, "A1__retrieve")
builder.add_edge("A1__retrieve", "generate_candidates")
builder.add_edge("generate_candidates", "Rank")
builder.add_edge("Rank", "merge_candidates")
builder.add_edge("merge_candidates", "SelectBest")
builder.add_edge("SelectBest", END)


def get_graph(checkpointer=None):
    """Build and return the compiled Algorithm 1 graph.

    Args:
        checkpointer: Optional checkpointer for state persistence.
            When provided, state is saved after every superstep. If None,
            a new checkpointer is created via get_checkpointer().
    """
    run_dir = get_run_dir()
    if checkpointer is None:
        checkpointer = get_checkpointer(run_dir)
    return builder.compile(checkpointer=checkpointer, name="Algorithm1")


def run(
    initial_state: Dict = None,
    run_dir: str = None,
    thread_id: str = None,
    session_id: str = None,
):
    """Run Algorithm 1 with the given configuration.

    Wraps the entire pipeline execution in a root SubgraphSpan so all
    traces appear grouped under a single "Algorithm1" trace in Langfuse.

    Creates a timestamped run directory and initializes a per-run MLELogger
    that writes JSON events to runs/{run_dir}/run.log.

    Args:
        initial_state: Optional initial state dict. If not provided,
            a default state with empty retrieved_models is used.
        run_dir: Optional run directory path. If not provided, auto-generates
            a timestamped directory under runs/ (e.g., runs/20260514103000_abc123f/).
        thread_id: Optional thread ID. Defaults to derived from run_dir basename.
        session_id: Optional session ID for grouping runs in Langfuse Sessions.

    Returns:
        Final state after graph execution.
    """
    if run_dir is None:
        run_dir = generate_run_dir()

    reset_llm_call_counter()
    os.makedirs(run_dir, exist_ok=True)
    run_id = os.path.basename(run_dir.rstrip("/"))

    run_dir_token = _current_run_dir.set(run_dir)
    run_id_token = _current_run_id.set(run_id)
    phase_token = _current_phase.set("search")

    mle_logger = MLELogger(run_dir=run_dir)
    logger_token = _current_mle_logger.set(mle_logger)

    if initial_state is None:
        initial_state = {
            "task_desc": "ML pipeline optimization",
            "metric_direction": "maximize",
            "score_function_desc": "",
            "retrieved_models": [],
            "model_descriptions": [],
            "datasets": [],
            "candidates_pool": [],
            "leaderboard": [],
            "best_candidate": {},
            "current_reference_idx": 0,
            "stage_history": [],
            "status": "start",
        }

    if thread_id is None:
        thread_id = get_thread_id(run_dir)

    if session_id is None:
        session_id = thread_id

    log_node_event(
        "Algorithm1",
        "run_start",
        {"task": initial_state.get("task_desc", ""), "run_dir": run_dir},
    )

    checkpointer = get_checkpointer(run_dir)
    config = {"configurable": {"thread_id": thread_id}}

    result = {}
    try:
        with SubgraphSpan(
            "Algorithm1",
            input_data={"task": initial_state.get("task_desc", ""), "status": "start"},
        ) as root_span:
            with propagate_attributes(session_id=session_id):
                graph = get_graph(checkpointer=checkpointer)
                result = graph.invoke(initial_state, config)
                root_span.set_output(
                    {
                        "status": result.get("status", ""),
                        "best_score": result.get("best_candidate", {}).get("score"),
                        "candidates": len(result.get("candidates_pool", [])),
                    }
                )
    finally:
        log_node_event("Algorithm1", "run_end", {"status": result.get("status", "")})
        mle_logger.close()
        _current_run_dir.reset(run_dir_token)
        _current_run_id.reset(run_id_token)
        _current_phase.reset(phase_token)
        _current_mle_logger.reset(logger_token)
        try:
            langfuse.flush()
        except Exception:
            log_node_event(
                "Algorithm1", "flush_error", {"error": "langfuse.flush() failed"}
            )

    return result
