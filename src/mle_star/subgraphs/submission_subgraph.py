"""Submission subgraph.

Implements the full submission flow (Section 5.6):
    START -> A_test__submit
          -> subsampling_extract
          -> subsampling_remove
          -> A12__check_leakage_submission --(fail)--> A12__fix_leakage_submission -> A12__check_leakage_submission
                                        |(ok)
                                    eval_submission -> END

Built in Stage 5 (mock). Real implementations in Stage 9.

Stage 10: Accepts interrupt_before and checkpointer params for
human-in-the-loop support. When interrupt_before=["eval_submission"],
the subgraph will pause before final submission evaluation.
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.mle_star.nodes.submission import (
    A_test__submit,
    subsampling_extract,
    subsampling_remove,
    A12__check_leakage_submission,
    A12__fix_leakage_submission,
    route_after_leakage_check_submission,
)
from src.mle_star.nodes.execution import eval_submission


class SubmissionState(TypedDict):
    """State for the submission subgraph.

    Contains the final solution code, submission script, and evaluation results.
    """

    final_solution: str
    best_score: float
    task_desc: str
    score_function_desc: str
    submission_code: str
    submission_score: float | None
    subsampling_block: str
    leakage_status: str | None
    final_dir: str
    status: str


_builder = StateGraph(SubmissionState)

_builder.add_node("A_test__submit", A_test__submit)
_builder.add_node("subsampling_extract", subsampling_extract)
_builder.add_node("subsampling_remove", subsampling_remove)
_builder.add_node("A12__check_leakage_submission", A12__check_leakage_submission)
_builder.add_node("A12__fix_leakage_submission", A12__fix_leakage_submission)
_builder.add_node("eval_submission", eval_submission)

_builder.add_edge(START, "A_test__submit")
_builder.add_edge("A_test__submit", "subsampling_extract")
_builder.add_edge("subsampling_extract", "subsampling_remove")
_builder.add_edge("subsampling_remove", "A12__check_leakage_submission")
_builder.add_conditional_edges(
    "A12__check_leakage_submission",
    route_after_leakage_check_submission,
    {
        "A12__fix_leakage_submission": "A12__fix_leakage_submission",
        "eval_submission": "eval_submission",
    },
)
_builder.add_edge("A12__fix_leakage_submission", "A12__check_leakage_submission")
_builder.add_edge("eval_submission", END)

_compiled_submission_subgraph = None


def get_submission_subgraph(
    interrupt_before: Optional[list[str]] = None,
    checkpointer=None,
):
    """Build and return the compiled submission subgraph.

    Topology:
        START -> A_test__submit -> subsampling_extract -> subsampling_remove
              -> A12__check_leakage_submission --(fail)--> A12__fix_leakage_submission -> A12__check_leakage_submission
                                                    |(ok)
                                              eval_submission -> END

    Args:
        interrupt_before: List of node names to interrupt before (e.g., ["eval_submission"]).
        checkpointer: LangGraph checkpointer for state persistence across interrupts.
    """
    global _compiled_submission_subgraph
    compile_kwargs = {"name": "SubmissionSubgraph"}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
        return _builder.compile(**compile_kwargs)
    if _compiled_submission_subgraph is None:
        _compiled_submission_subgraph = _builder.compile(**compile_kwargs)
    return _compiled_submission_subgraph
