"""Ensemble round subgraph for Algorithm 3.

Implements a SINGLE-PASS ensemble round: A9→A10→A12_check→eval_ensemble
with debug retry and leakage fix conditional edges:
    A12__check_leakage_ensemble --(fail)--> A12__fix_leakage_ensemble --> A12__check_leakage_ensemble
    A12__check_leakage_ensemble --(ok)--> eval_ensemble
    eval_ensemble --(error, retries < MAX)--> A11__debug_ensemble --> A10__implement_ensemble
    eval_ensemble --(ok)--> END
    eval_ensemble --(error, retries >= MAX)--> END

The R-round iteration loop is handled by the Python-level
loop in algorithm_3.py, each round wrapped in a SubgraphSpan for clean
Langfuse traces. This matches the refinement_subgraph pattern (single-pass
step subgraph invoked repeatedly by the outer Python loop).
"""

from langgraph.graph import END, START, StateGraph

from src.mle_star.state.alg3_state import EnsembleRoundState
from src.mle_star.nodes.ensemble import (
    A9__plan_ensemble,
    A10__implement_ensemble,
)
from src.mle_star.nodes.robustness import (
    A11__debug_ensemble,
    A12__check_leakage_ensemble,
    A12__fix_leakage_ensemble,
    route_after_leakage_check_ensemble,
    route_after_ensemble_eval,
)
from src.mle_star.nodes.execution import eval_ensemble


_round_builder = StateGraph(EnsembleRoundState)

_round_builder.add_node("A9__plan_ensemble", A9__plan_ensemble)
_round_builder.add_node("A10__implement_ensemble", A10__implement_ensemble)
_round_builder.add_node("A12__check_leakage_ensemble", A12__check_leakage_ensemble)
_round_builder.add_node("A12__fix_leakage_ensemble", A12__fix_leakage_ensemble)
_round_builder.add_node("eval_ensemble", eval_ensemble)
_round_builder.add_node("A11__debug_ensemble", A11__debug_ensemble)

_round_builder.add_edge(START, "A9__plan_ensemble")
_round_builder.add_edge("A9__plan_ensemble", "A10__implement_ensemble")
_round_builder.add_edge("A10__implement_ensemble", "A12__check_leakage_ensemble")
_round_builder.add_conditional_edges(
    "A12__check_leakage_ensemble",
    route_after_leakage_check_ensemble,
    {
        "A12__fix_leakage_ensemble": "A12__fix_leakage_ensemble",
        "eval_ensemble": "eval_ensemble",
    },
)
_round_builder.add_edge("A12__fix_leakage_ensemble", "A12__check_leakage_ensemble")
_round_builder.add_conditional_edges(
    "eval_ensemble",
    route_after_ensemble_eval,
    {
        "A11__debug_ensemble": "A11__debug_ensemble",
        END: END,
    },
)
_round_builder.add_edge("A11__debug_ensemble", "A10__implement_ensemble")

_compiled_round_subgraph = None


def get_ensemble_round_subgraph():
    """Build and return the compiled single-pass ensemble round subgraph.

    Topology (one round only — the R-iteration loop is handled in algorithm_3.py):
        START -> A9__plan_ensemble
              -> A10__implement_ensemble
              -> A12__check_leakage_ensemble --(fail)--> A12__fix_leakage_ensemble -> A12__check_leakage_ensemble
                                           |(ok)
              eval_ensemble
                 |--(error, debug_retries < MAX)--> A11__debug_ensemble -> A10__implement_ensemble
                 |--(ok)--> END
                 |--(error, debug_retries >= MAX)--> END
    """
    global _compiled_round_subgraph
    if _compiled_round_subgraph is None:
        _compiled_round_subgraph = _round_builder.compile(name="EnsembleRoundSubgraph")
    return _compiled_round_subgraph
