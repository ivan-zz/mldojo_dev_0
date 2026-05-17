"""State schemas for Algorithm 1 (Search phase).

Contains Alg1State, CandidateState, MergeState, FanoutState, and MergeFanoutState.
Stage 7 additions: model_descriptions, score_function_desc, datasets, task_desc
propagated through subgraph states for real LLM calls and code execution.
"""

import operator
from typing import Annotated, Dict, List, Optional, TypedDict


class Alg1State(TypedDict):
    """Parent graph state for Algorithm 1 (Search phase).

    Attributes:
        task_desc: Description of the ML task.
        metric_direction: 'maximize' or 'minimize' — controls score comparison.
        score_function_desc: Description of the evaluation metric (e.g., 'RMSLE').
        retrieved_models: List of model names retrieved for evaluation.
        model_descriptions: List of model description dicts from A1__retrieve.
        datasets: List of dataset file paths for execution.
        candidates_pool: Pool of candidate solutions (with operator.add reducer).
        leaderboard: Ranked list of candidates by score.
        best_candidate: Current best candidate solution.
        current_reference_idx: Index for merge reference iteration.
        stage_history: List of all trace events from node execution.
        status: Current status of the pipeline ('start', 'done', etc.).
    """

    task_desc: str
    metric_direction: str
    score_function_desc: str
    retrieved_models: List[str]
    model_descriptions: List[Dict]
    datasets: List[str]
    candidates_pool: Annotated[List[Dict], operator.add]
    leaderboard: List[Dict]
    best_candidate: Dict
    current_reference_idx: int
    stage_history: List[Dict]
    status: str


class CandidateState(TypedDict):
    """State for candidate generation subgraph (per model).

    Each model gets its own subgraph instance with this state.

    Attributes:
        model: Name of the model being evaluated (e.g., 'RandomForestRegressor').
        model_description: Dict with model details from A1__retrieve
            ({model_name, description, example_code}).
        task_desc: Description of the ML task (for prompt context).
        score_function_desc: Description of the evaluation metric.
        datasets: List of dataset file paths for execution.
        metric_direction: 'maximize' or 'minimize'.
        code: Generated code string for the candidate.
        score: Evaluation score.
        attempts: Number of debug/retry attempts made.
        usage_fix_attempts: Number of data usage fix attempts.
        leakage_fix_attempts: Number of data leakage fix attempts.
        execution_output: Stdout from subprocess execution.
        execution_error: Stderr from subprocess execution, or None.
        sub_events: List of sub-events from retry loops.
        status: Current status ('pending', 'ok', 'usage_fail', 'leakage_fail', 'crashed', 'error').
    """

    model: str
    model_description: Optional[Dict]
    task_desc: str
    score_function_desc: str
    datasets: List[str]
    metric_direction: str
    code: str
    score: float
    attempts: int
    usage_fix_attempts: int
    leakage_fix_attempts: int
    execution_output: str
    execution_error: Optional[str]
    sub_events: List[Dict]
    status: str


class MergeState(TypedDict):
    """State for merge subgraph (per merge pair).

    Attributes:
        base_code: Code of the best candidate to merge from.
        ref_code: Code of the reference candidate to merge with.
        merged_code: Result of merging base_code and ref_code.
        task_desc: Description of the ML task (for prompt context).
        score_function_desc: Description of the evaluation metric.
        datasets: List of dataset file paths for execution.
        metric_direction: 'maximize' or 'minimize'.
        score: Evaluation score of the merged candidate.
        attempts: Number of debug/retry attempts made.
        leakage_fix_attempts: Number of data leakage fix attempts.
        execution_output: Stdout from subprocess execution.
        execution_error: Stderr from subprocess execution, or None.
        sub_events: List of sub-events from retry loops.
        status: Current status ('pending', 'ok', 'leakage_fail', 'crashed', 'error').
    """

    base_code: str
    ref_code: str
    merged_code: str
    task_desc: str
    score_function_desc: str
    datasets: List[str]
    metric_direction: str
    score: float
    attempts: int
    leakage_fix_attempts: int
    execution_output: str
    execution_error: Optional[str]
    sub_events: List[Dict]
    status: str


class FanoutState(TypedDict):
    """State for candidate fan-out subgraph (generating all candidates).

    Attributes:
        retrieved_models: List of model names to generate candidates for.
        model_descriptions: List of model description dicts.
        task_desc: Description of the ML task.
        score_function_desc: Description of the evaluation metric.
        datasets: List of dataset file paths.
        metric_direction: 'maximize' or 'minimize'.
        candidates_pool: Accumulated candidate results (reduced via operator.add).
    """

    retrieved_models: List[str]
    model_descriptions: List[Dict]
    task_desc: str
    score_function_desc: str
    datasets: List[str]
    metric_direction: str
    candidates_pool: Annotated[List[Dict], operator.add]


class MergeFanoutState(TypedDict):
    """State for merge fan-out subgraph (merging all pairs).

    Attributes:
        best_candidate: The best candidate from ranking.
        leaderboard: Ranked list of candidates.
        task_desc: Description of the ML task.
        score_function_desc: Description of the evaluation metric.
        datasets: List of dataset file paths.
        metric_direction: 'maximize' or 'minimize'.
        candidates_pool: Accumulated merge results (reduced via operator.add).
    """

    best_candidate: Dict
    leaderboard: List[Dict]
    task_desc: str
    score_function_desc: str
    datasets: List[str]
    metric_direction: str
    candidates_pool: Annotated[List[Dict], operator.add]
