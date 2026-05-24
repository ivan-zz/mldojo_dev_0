"""State schema for Algorithm 2 (Ablation + Refinement).

Contains Alg2State (outer loop), AblationCycleState (ablation subgraph),
AblationVariantState (per-variant execution subgraph), and
RefinementInnerState (refinement inner loop subgraph).

The outer loop is Python-level (not LangGraph conditional edges), so Alg2State
does not use operator.add — lists are extended manually after each cycle.
The narrower subgraph states use operator.add only for Send API fan-out
aggregation within a single ablation cycle.
"""

import operator
from typing import Annotated, TypedDict


class Alg2State(TypedDict):
    """State for Algorithm 2 (Ablation + Refinement).

    Outer-loop state managed by a Python-level for loop in
    algorithms/algorithm_2.py. No operator.add — lists extended manually.
    """

    # ── Input ──
    current_solution: str
    best_score: float
    metric_direction: str

    # ── Ablation ──
    ablation_scripts: list[dict]
    ablation_results_list: list[dict]
    ablation_summaries: list[str]
    target_block: str
    initial_plan: str
    refined_blocks: list[str]

    # ── Refinement Inner Loop ──
    current_plans: list[str]
    current_scores: list[float]
    refined_code: str
    candidate_solution: str

    # ── Execution ──
    execution_output: str
    execution_error: str | None
    execution_score: float | None

    # ── Control ──
    outer_step: int
    inner_step: int

    # ── Robustness ──
    leakage_status: str | None
    leakage_code_block: str | None
    debug_history: list[dict]
    security_violations: list[str]

    # ── Output ──
    improved_solution: str
    improved_score: float
    convergence_achieved: bool

    stage_history: list[dict]
    status: str


class AblationCycleState(TypedDict):
    """State for a single ablation cycle (single-pass subgraph).

    Used by subgraphs/ablation_subgraph.py. Narrow state avoids
    operator.add double-accumulation when merged back into Alg2State.
    Only ablation_results_list uses operator.add for Send API fan-out.
    """

    current_solution: str
    best_score: float
    metric_direction: str
    ablation_scripts: list[dict]
    functional_blocks: list[dict]
    ablation_results_list: Annotated[list[dict], operator.add]
    previous_summaries: list[str]
    previous_blocks: list[str]
    ablation_summaries: list[str]
    target_block: str
    initial_plan: str
    status: str


class AblationFanoutState(TypedDict):
    """Reserved: State for the ablation Send API fan-out.

    Not currently used — the ablation subgraph dispatches directly via
    AblationVariantState dicts in _dispatch_from_cycle_state. Retained
    for potential future use when ablation fan-out is a compiled subgraph.

    Each variant execution runs with one ablation script. Results are
    aggregated via operator.add.
    """

    current_solution: str
    ablation_scripts: list[dict]
    ablation_results: Annotated[list[dict], operator.add]


class AblationVariantState(TypedDict):
    """State for executing a single ablation variant (baseline or disabled component).

    Includes debug retry loop (max 3 attempts) for failed executions.
    Used by subgraphs/ablation_variant_subgraph.py.
    """

    variant_name: str
    variant_code: str
    block_name: str
    execution_output: str
    execution_error: str | None
    execution_exit_code: int
    execution_score: float | None
    attempts: int
    status: str


class RefinementInnerState(TypedDict):
    """State for the refinement inner loop subgraph.

    Used by subgraphs/refinement_subgraph.py. Narrow state that only
    tracks refinement-relevant fields, avoiding operator.add issues.
    """

    target_block: str
    current_solution: str
    best_score: float
    metric_direction: str
    initial_plan: str
    current_plan: str
    refined_code: str
    candidate_solution: str
    execution_output: str
    execution_error: str | None
    execution_score: float | None
    inner_step: int
    debug_retries: int
    improved_solution: str
    improved_score: float
    status: str
