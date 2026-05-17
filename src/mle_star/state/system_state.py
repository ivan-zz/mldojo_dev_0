"""System state for the MLE-STAR overarching graph.

Contains MleStarSystemState — the top-level state that routes between
Algorithm 1 (search), Algorithm 2 (ablation+refinement), and
Algorithm 3 (ensemble) phases.
"""

import operator
from typing import Annotated, TypedDict


class MleStarSystemState(TypedDict):
    """Overarching state for the MLE-STAR main graph.

    Controls phase transitions and carries shared data across Algorithm
    subgraphs. All accumulation fields use Annotated[list, operator.add].
    """

    # ── Input ──
    task_desc: str
    datasets: list[str]
    score_function_desc: str

    # ── Phase Control ──
    phase: str
    phase_history: Annotated[list[dict], operator.add]

    # ── Solutions ──
    current_solution: str
    best_solution: str
    best_score: float
    current_score: float | None

    # ── Metric Direction ──
    metric_direction: str
    raw_best_score: float | None

    # ── Search Phase Output ──
    alg1_result: dict

    # ── Ablation+Refinement Phase ──
    alg2_result: dict
    outer_step: int
    inner_step: int
    convergence_achieved: bool

    # ── Parallel Pipeline Results (D15) ──
    parallel_results: Annotated[list[dict], operator.add]

    # ── Ensemble Phase ──
    alg3_result: dict
    ensemble_solutions: list[str]
    ensemble_input_scores: list[float]
    ensemble_round: int

    # ── Submission ──
    submission_code: str
    submission_score: float | None
    final_dir: str

    # ── Stagnation Control ──
    full_cycles: int
    max_full_cycles: int

    # ── Robustness ──
    debug_history: Annotated[list[dict], operator.add]
    security_violations: Annotated[list[str], operator.add]

    # ── Status ──
    stage_history: Annotated[list[dict], operator.add]
    status: str
