"""State schemas for parallel pipeline fan-out (D15).

Contains PipelineFlowState and ParallelFanoutState used by
subgraphs/parallel_pipeline_subgraph.py for L independent (Alg1+Alg2)
pipeline runs via Send API fan-out.
"""

import operator
from typing import Annotated, TypedDict


class PipelineFlowState(TypedDict):
    """Per-pipeline state passed via Send API to pipeline_flow_node.

    Each instance runs a full Alg1+Alg2 pipeline independently,
    producing a distinct candidate solution for ensemble input.
    """

    run_index: int
    task_desc: str
    datasets: list[str]
    score_function_desc: str
    metric_direction: str


class ParallelFanoutState(TypedDict):
    """Fan-out wrapper state for parallel pipeline dispatch.

    Uses operator.add to accumulate parallel_results from L
    independent pipeline runs, matching the Send API fan-out pattern
    from algorithm_1.py's FanoutState.
    """

    task_desc: str
    datasets: list[str]
    score_function_desc: str
    metric_direction: str
    num_parallel_solutions: int
    parallel_results: Annotated[list[dict], operator.add]
