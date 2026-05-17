"""Ablation subgraph for a single outer-loop cycle (single-pass topology).

Implements the ablation flow with Send API fan-out for parallel variant execution:
    A4__generate_ablation → Send API fan-out (one per variant script)
        → ablation_variant_flow (baseline + ablation variants)
    → A5__summarize_ablation (aggregates all variant results)
    → A6__extract_block (picks most impactful block, generates p_0)
    → END

Each variant is executed independently via the ablation_variant_subgraph,
which includes a debug retry loop (max 3 attempts) for failed executions.

Single-pass: No outer loop conditional edges. The outer loop is managed
by the Python-level for loop in algorithms/algorithm_2.py.
"""

from typing import List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.mle_star.state.alg2_state import AblationCycleState
from src.mle_star.state.shared import SubgraphSpan, log_node_event
from src.mle_star.nodes.ablation import (
    A4__generate_ablation,
    A5__summarize_ablation,
    A6__extract_block,
)
from src.mle_star.subgraphs.ablation_variant_subgraph import (
    ablation_variant_subgraph,
)


def _dispatch_from_cycle_state(state: AblationCycleState) -> List[Send]:
    """Bridge: extract ablation_scripts from cycle state for Send API dispatch.

    A4 generates ablation_scripts: [{name, code, block_name}].
    This function dispatches one ablation_variant_flow per script.
    """
    scripts = state.get("ablation_scripts", [])
    log_node_event(
        "A4__dispatch",
        "dispatch",
        {
            "num_variants": len(scripts),
            "variant_names": [s.get("name") for s in scripts],
        },
    )
    sends = []
    for script in scripts:
        sends.append(
            Send(
                "ablation_variant_flow",
                {
                    "variant_name": script.get("name", "unknown"),
                    "variant_code": script.get("code", ""),
                    "block_name": script.get("block_name", "unknown"),
                    "execution_output": "",
                    "execution_error": None,
                    "execution_score": None,
                    "attempts": 0,
                    "status": "pending",
                },
            )
        )
    return sends


def ablation_variant_flow_node(state: dict) -> dict:
    """Invoke ablation variant subgraph for one script.

    Runs the full eval -> debug retry flow for a single ablation variant.
    Wrapped in SubgraphSpan for grouped tracing.

    Returns aggregated ablation results for collection via operator.add.
    """
    variant_name = state.get("variant_name", "unknown")
    block_name = state.get("block_name", "unknown")

    with SubgraphSpan(
        "ablation_variant_subgraph",
        input_data={
            "variant_name": variant_name,
            "block_name": block_name,
            "status": "pending",
        },
    ) as span:
        sub_state = dict(state)
        result = ablation_variant_subgraph.invoke(sub_state, span.config)
        span.set_output(
            {
                "variant_name": variant_name,
                "block_name": block_name,
                "execution_score": result.get("execution_score"),
                "status": result.get("status"),
            }
        )

    baseline_score = state.get("best_score", 0) or 0.85
    execution_score = result.get("execution_score") or 0
    impact = round(max(0, baseline_score - execution_score), 4)

    return {
        "ablation_results_list": [
            {
                "variant_name": variant_name,
                "block_name": block_name,
                "execution_score": execution_score,
                "impact": impact,
                "status": result.get("status"),
                "attempts": result.get("attempts", 0),
            }
        ]
    }


_builder = StateGraph(AblationCycleState)

_builder.add_node("A4__generate_ablation", A4__generate_ablation)
_builder.add_node("ablation_variant_flow", ablation_variant_flow_node)
_builder.add_node("A5__summarize_ablation", A5__summarize_ablation)
_builder.add_node("A6__extract_block", A6__extract_block)

_builder.add_edge(START, "A4__generate_ablation")
_builder.add_conditional_edges(
    "A4__generate_ablation",
    _dispatch_from_cycle_state,
    ["ablation_variant_flow"],
)
_builder.add_edge("ablation_variant_flow", "A5__summarize_ablation")
_builder.add_edge("A5__summarize_ablation", "A6__extract_block")
_builder.add_edge("A6__extract_block", END)


def get_ablation_subgraph():
    """Build and return the compiled ablation subgraph (single-pass).

    Topology:
        START -> A4__generate_ablation
                  | (produces ablation_scripts)
                  |
             dispatch (Send API fan-out)
                  |
                  |-- ablation_variant_flow (baseline)
                  |     |-- ablation_variant_subgraph (eval + debug retry)
                  |-- ablation_variant_flow (ablation_0)
                  |-- ablation_variant_flow (ablation_1)
                  |
             A5__summarize_ablation
                  |
             A6__extract_block
                  |
             END
    """
    return _builder.compile(name="AblationSubgraph")
