"""Refinement step subgraph for Algorithm 2 inner loop.

Implements a SINGLE-PASS step: A7→A_verify→A_sast→eval_refinement
with debug retry conditional edges only:
    eval_refinement → (error, debug_retries < MAX) → A11__debug_refine → A7__implement
    eval_refinement → (ok) → END
    eval_refinement → (error, debug_retries >= MAX) → END

The inner loop (K refinement iterations) is handled by the Python-level
loop in algorithm_2.py, each step wrapped in a SubgraphSpan for clean
Langfuse traces.

The old get_refinement_subgraph() with A8__plan conditional edges is
replaced by get_refinement_step_subgraph() which does ONE step only.
A8__plan is no longer inside this subgraph — the Python loop in
algorithm_2.py handles plan updates between steps.

Stage 10: Accepts interrupt_before and checkpointer params for
human-in-the-loop support. When interrupt_before=["A_sast"], the
subgraph will pause before A_sast, allowing human review of code
before security scanning.
"""

from typing import Optional

from langgraph.graph import END, START, StateGraph

from src.mle_star.state.alg2_state import RefinementInnerState
from src.mle_star.nodes.verification import A_verify, A_sast
from src.mle_star.nodes.refinement import (
    A7__implement,
    eval_refinement,
    A11__debug_refine,
    route_after_verify,
    route_after_sast,
    route_after_eval_step,
)


_step_builder = StateGraph(RefinementInnerState)

_step_builder.add_node("A7__implement", A7__implement)
_step_builder.add_node("A_verify", A_verify)
_step_builder.add_node("A_sast", A_sast)
_step_builder.add_node("eval_refinement", eval_refinement)
_step_builder.add_node("A11__debug_refine", A11__debug_refine)

_step_builder.add_edge(START, "A7__implement")
_step_builder.add_edge("A7__implement", "A_verify")
_step_builder.add_conditional_edges(
    "A_verify",
    route_after_verify,
    {"A7__implement": "A7__implement", "A_sast": "A_sast"},
)
_step_builder.add_conditional_edges(
    "A_sast",
    route_after_sast,
    {"A7__implement": "A7__implement", "eval_refinement": "eval_refinement"},
)
_step_builder.add_conditional_edges(
    "eval_refinement",
    route_after_eval_step,
    {
        "A11__debug_refine": "A11__debug_refine",
        END: END,
    },
)
_step_builder.add_edge("A11__debug_refine", "A7__implement")

_compiled_step_subgraph = None


def get_refinement_step_subgraph(
    interrupt_before: Optional[list[str]] = None,
    checkpointer=None,
):
    """Build and return the compiled single-pass refinement step subgraph.

    Topology (one step only — the K-iteration loop is handled in algorithm_2.py):
        START → A7__implement
              → A_verify --(fail)--> A7__implement (with feedback)
                    |(pass)
              A_sast --(critical)--> A7__implement (with report)
                    |(pass)
              eval_refinement
                    |--(error, debug_retries < MAX)--> A11__debug_refine → A7__implement
                    |--(ok)--> END
                    |--(error, debug_retries >= MAX)--> END

    Args:
        interrupt_before: List of node names to interrupt before (e.g., ["A_sast"]).
        checkpointer: LangGraph checkpointer for state persistence across interrupts.
    """
    global _compiled_step_subgraph
    compile_kwargs = {"name": "RefinementStepSubgraph"}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
        return _step_builder.compile(**compile_kwargs)
    if _compiled_step_subgraph is None:
        _compiled_step_subgraph = _step_builder.compile(**compile_kwargs)
    return _compiled_step_subgraph


def get_refinement_subgraph(
    interrupt_before: Optional[list[str]] = None,
    checkpointer=None,
):
    """Backward-compatible alias — returns the step subgraph.

    The old multi-step refinement subgraph with A8__plan conditional edges
    has been replaced by the single-pass step subgraph. The Python-level
    loop in algorithm_2.py handles the K-iteration refinement cycle.
    """
    return get_refinement_step_subgraph(
        interrupt_before=interrupt_before, checkpointer=checkpointer
    )
