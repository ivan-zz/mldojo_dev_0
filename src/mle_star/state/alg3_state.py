"""State schema for Algorithm 3 (Ensemble).

Contains Alg3State (outer loop state) and EnsembleRoundState (single-round
subgraph state). The R-round ensemble loop is managed by a Python-level
for loop in algorithm_3.py. Each round invokes the ensemble_round_subgraph
with EnsembleRoundState.
"""

import operator
from typing import Annotated, TypedDict


class Alg3State(TypedDict):
    """State for Algorithm 3 (Ensemble).

    Outer-loop state managed by a Python-level for loop in
    algorithms/algorithm_3.py. No operator.add — lists extended manually.
    """

    # ── Input ──
    ensemble_solutions: list[str]
    ensemble_input_scores: list[float]
    metric_direction: str

    # ── Ensemble Loop ──
    ensemble_plans: list[str]
    ensemble_scores: list[float]
    current_ensemble_plan: str
    current_ensemble_code: str
    current_ensemble_score: float
    ensemble_round: int

    # ── Execution ──
    execution_output: str
    execution_error: str | None
    execution_score: float | None

    # ── Robustness ──
    debug_history: Annotated[list[dict], operator.add]
    leakage_status: str | None
    leakage_code_block: str | None
    debug_retries: int

    # ── Output ──
    best_ensemble_code: str
    best_ensemble_score: float

    stage_history: Annotated[list[dict], operator.add]
    status: str


class EnsembleRoundState(TypedDict):
    """Narrow state for a single ensemble round subgraph pass.

    Used by subgraphs/ensemble_round_subgraph.py. Contains only the
    fields relevant to a single A9→A10→A12→eval→debug retry cycle.
    The R-round loop is managed by algorithm_3.py at the Python level.
    """

    ensemble_solutions: list[str]
    ensemble_input_scores: list[float]
    metric_direction: str
    current_ensemble_plan: str
    current_ensemble_code: str
    current_ensemble_score: float
    ensemble_round: int

    execution_output: str
    execution_error: str | None
    execution_score: float | None

    debug_retries: int
    leakage_status: str | None
    leakage_code_block: str | None

    best_ensemble_code: str
    best_ensemble_score: float
    status: str
