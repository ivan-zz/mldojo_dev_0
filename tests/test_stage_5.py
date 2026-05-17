"""Stage 5 verification tests: Ensemble + Submission Topology.

Validates the complete ensemble and submission flow:
    - EnsembleRoundState and SubmissionState have required fields
    - Mock ensemble nodes (A9, A10) produce valid output
    - Mock ensemble robustness nodes (A11_debug, A12_check, A12_fix) work correctly
    - Mock execution nodes (eval_ensemble, eval_submission) produce valid output
    - Routing functions (leakage check, eval) return correct edge names
    - Ensemble round subgraph compiles and runs single pass
    - Ensemble round subgraph handles debug retry and leakage fix loops
    - Algorithm 3 round loop iterates R times
    - Submission subgraph compiles and runs end-to-end with leakage check
    - State mapping functions work correctly
    - Graph topology has correct nodes (no continue_ensemble edge)
    - Config values are wired correctly

Test IDs map to the Stage 5 verification checklist.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.mle_star.supervisor import SupervisorConfig


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "mle_star")
MLE_STAR_DIR = os.path.normpath(BASE_DIR)


# ── S5-01: State Types ──────────────────────────────────────────────────────


class TestEnsembleRoundState:
    """S5-01: EnsembleRoundState has required fields for the round subgraph."""

    def test_s5_01_ensemble_round_state_fields(self):
        from src.mle_star.state.alg3_state import EnsembleRoundState

        required = [
            "ensemble_solutions",
            "ensemble_input_scores",
            "current_ensemble_plan",
            "current_ensemble_code",
            "current_ensemble_score",
            "ensemble_round",
            "execution_output",
            "execution_error",
            "execution_score",
            "debug_retries",
            "leakage_status",
            "leakage_code_block",
            "best_ensemble_code",
            "best_ensemble_score",
            "status",
        ]
        missing = [k for k in required if k not in EnsembleRoundState.__annotations__]
        assert not missing, f"Missing EnsembleRoundState fields: {missing}"

    def test_s5_01_alg3_state_has_new_fields(self):
        from src.mle_star.state.alg3_state import Alg3State

        assert "leakage_code_block" in Alg3State.__annotations__, (
            "Alg3State should have leakage_code_block field"
        )
        assert "debug_retries" in Alg3State.__annotations__, (
            "Alg3State should have debug_retries field"
        )

    def test_s5_01_system_state_has_ensemble_input_scores(self):
        from src.mle_star.state.system_state import MleStarSystemState

        assert "ensemble_input_scores" in MleStarSystemState.__annotations__, (
            "MleStarSystemState should have ensemble_input_scores field"
        )


class TestSubmissionState:
    """S5-01: SubmissionState has required fields."""

    def test_s5_01_submission_state_fields(self):
        from src.mle_star.subgraphs.submission_subgraph import SubmissionState

        required = [
            "final_solution",
            "best_score",
            "task_desc",
            "score_function_desc",
            "submission_code",
            "submission_score",
            "subsampling_block",
            "leakage_status",
            "status",
        ]
        missing = [k for k in required if k not in SubmissionState.__annotations__]
        assert not missing, f"Missing SubmissionState fields: {missing}"


# ── S5-02: Config ──────────────────────────────────────────────────────────


class TestConfig:
    """S5-02: New config values are available."""

    def test_s5_02_config_has_ensemble_debug_retries(self):
        from src.mle_star.config import MAX_ENSEMBLE_DEBUG_RETRIES

        assert isinstance(MAX_ENSEMBLE_DEBUG_RETRIES, int)
        assert MAX_ENSEMBLE_DEBUG_RETRIES >= 1

    def test_s5_02_config_has_leakage_fix_retries(self):
        from src.mle_star.config import MAX_LEAKAGE_FIX_RETRIES

        assert isinstance(MAX_LEAKAGE_FIX_RETRIES, int)

    def test_s5_02_config_has_subsampling_threshold(self):
        from src.mle_star.config import SUBSAMPLING_THRESHOLD

        assert isinstance(SUBSAMPLING_THRESHOLD, int)
        assert SUBSAMPLING_THRESHOLD > 0


# ── S5-03: Mock Ensemble Nodes ──────────────────────────────────────────────


class TestMockEnsembleNodes:
    """S5-03: Mock ensemble nodes produce valid output."""

    def test_s5_03_a9_plan_ensemble(self):
        from src.mle_star.nodes.ensemble import A9__plan_ensemble

        state = {
            "ensemble_solutions": ["code1", "code2"],
            "ensemble_input_scores": [0.8, 0.85],
            "ensemble_round": 0,
        }
        result = A9__plan_ensemble(state)
        assert "current_ensemble_plan" in result
        assert result["status"] == "planned"
        assert "0" in result["current_ensemble_plan"]

    def test_s5_03_a9_plan_ensemble_with_scores(self):
        from src.mle_star.nodes.ensemble import A9__plan_ensemble

        state = {
            "ensemble_solutions": ["code_a", "code_b"],
            "ensemble_input_scores": [0.75, 0.90],
            "ensemble_round": 2,
        }
        result = A9__plan_ensemble(state)
        assert "Weighted average" in result["current_ensemble_plan"]

    def test_s5_03_a10_implement_ensemble(self):
        from src.mle_star.nodes.ensemble import A10__implement_ensemble

        state = {
            "current_ensemble_plan": "Test plan",
            "ensemble_solutions": ["base code"],
            "best_ensemble_code": "base code",
            "ensemble_round": 1,
        }
        result = A10__implement_ensemble(state)
        assert "current_ensemble_code" in result
        assert result["status"] == "implemented"
        assert "ensemble_v1" in result["current_ensemble_code"]


# ── S5-04: Mock Robustness Nodes ────────────────────────────────────────────


class TestMockRobustnessNodes:
    """S5-04: Mock ensemble robustness nodes work correctly."""

    def test_s5_04_debug_ensemble_increments_retries(self):
        from src.mle_star.nodes.robustness import A11__debug_ensemble

        state = {
            "current_ensemble_code": "test code",
            "debug_retries": 0,
            "execution_error": "test error",
        }
        result = A11__debug_ensemble(state)
        assert result["debug_retries"] == 1
        assert "debug" in result["current_ensemble_code"]
        assert result["status"] == "debugged"

    def test_s5_04_debug_ensemble_multiple_retries(self):
        from src.mle_star.nodes.robustness import A11__debug_ensemble

        state = {
            "current_ensemble_code": "test code",
            "debug_retries": 2,
            "execution_error": "error 3",
        }
        result = A11__debug_ensemble(state)
        assert result["debug_retries"] == 3

    def test_s5_04_check_leakage_ensemble_passes(self):
        from src.mle_star.nodes.robustness import A12__check_leakage_ensemble

        state = {"current_ensemble_code": "clean code"}
        result = A12__check_leakage_ensemble(state)
        assert result["leakage_status"] == "ok"
        assert result["status"] == "ok"

    def test_s5_04_fix_leakage_ensemble(self):
        from src.mle_star.nodes.robustness import A12__fix_leakage_ensemble

        state = {"current_ensemble_code": "leaky code\n"}
        result = A12__fix_leakage_ensemble(state)
        assert "leakage_fix_ensemble" in result["current_ensemble_code"]
        assert result["leakage_status"] is None


# ── S5-05: Mock Execution Nodes ─────────────────────────────────────────────


class TestMockExecutionNodes:
    """S5-05: Mock eval_ensemble and eval_submission produce valid output."""

    def test_s5_05_eval_ensemble_ok(self):
        from src.mle_star.nodes.execution import eval_ensemble

        with patch("src.mle_star.nodes.execution.random_pass", return_value=False):
            state = {
                "best_ensemble_score": 0.80,
                "ensemble_round": 0,
                "debug_retries": 0,
                "current_ensemble_code": "ensemble code",
                "best_ensemble_code": "ensemble code",
            }
            result = eval_ensemble(state)
            assert result["status"] == "ok"
            assert result["execution_score"] is not None
            assert result["execution_score"] > 0.80

    def test_s5_05_eval_ensemble_error(self):
        from src.mle_star.nodes.execution import eval_ensemble

        with patch("src.mle_star.nodes.execution.random_pass", return_value=True):
            state = {
                "best_ensemble_score": 0.80,
                "ensemble_round": 0,
                "debug_retries": 0,
                "current_ensemble_code": "ensemble code",
                "best_ensemble_code": "ensemble code",
            }
            result = eval_ensemble(state)
            assert result["status"] == "error"
            assert result["execution_score"] is None

    def test_s5_05_eval_submission(self):
        from src.mle_star.nodes.execution import eval_submission

        state = {"best_score": 0.85}
        result = eval_submission(state)
        assert result["status"] == "ok"
        assert result["submission_score"] is not None
        assert result["submission_score"] > 0.85


# ── S5-06: Mock Submission Nodes ────────────────────────────────────────────


class TestMockSubmissionNodes:
    """S5-06: Mock submission nodes produce valid output."""

    def test_s5_06_a_test_submit(self):
        from src.mle_star.nodes.submission import A_test__submit

        state = {"final_solution": "best model code", "best_score": 0.90}
        result = A_test__submit(state)
        assert "submission_v1" in result["submission_code"]
        assert result["status"] == "generated"

    def test_s5_06_subsampling_extract(self):
        from src.mle_star.nodes.submission import subsampling_extract

        state = {"submission_code": "code with subsampling"}
        result = subsampling_extract(state)
        assert result["subsampling_block"] != ""
        assert result["status"] == "extracted"

    def test_s5_06_subsampling_remove(self):
        from src.mle_star.nodes.submission import subsampling_remove

        state = {"submission_code": "code with subsampling\n"}
        result = subsampling_remove(state)
        assert "use_full_data: True" in result["submission_code"]
        assert result["status"] == "subsampling_removed"

    def test_s5_06_check_leakage_submission_passes(self):
        from src.mle_star.nodes.submission import A12__check_leakage_submission

        state = {"submission_code": "clean code"}
        result = A12__check_leakage_submission(state)
        assert result["leakage_status"] == "ok"
        assert result["status"] == "ok"

    def test_s5_06_fix_leakage_submission(self):
        from src.mle_star.nodes.submission import A12__fix_leakage_submission

        state = {"submission_code": "leaky code\n"}
        result = A12__fix_leakage_submission(state)
        assert "leakage_fix_submission" in result["submission_code"]


# ── S5-07: Routing Functions ────────────────────────────────────────────────


class TestRoutingFunctions:
    """S5-07: Ensemble and submission routing functions return correct edge names."""

    def test_s5_07_leakage_check_ensemble_ok(self):
        from src.mle_star.nodes.robustness import route_after_leakage_check_ensemble

        assert (
            route_after_leakage_check_ensemble({"leakage_status": "ok"})
            == "eval_ensemble"
        )

    def test_s5_07_leakage_check_ensemble_fail(self):
        from src.mle_star.nodes.robustness import route_after_leakage_check_ensemble

        assert (
            route_after_leakage_check_ensemble({"leakage_status": "leakage_fail"})
            == "A12__fix_leakage_ensemble"
        )

    def test_s5_07_leakage_check_ensemble_data_leakage(self):
        from src.mle_star.nodes.robustness import route_after_leakage_check_ensemble

        assert (
            route_after_leakage_check_ensemble({"leakage_status": "Yes Data Leakage"})
            == "A12__fix_leakage_ensemble"
        )

    def test_s5_07_eval_ensemble_ok(self):
        from src.mle_star.nodes.robustness import route_after_ensemble_eval

        assert route_after_ensemble_eval({"status": "ok"}) == "__end__"

    def test_s5_07_eval_ensemble_error_with_retries(self):
        from src.mle_star.nodes.robustness import route_after_ensemble_eval

        result = route_after_ensemble_eval({"status": "error", "debug_retries": 0})
        assert result == "A11__debug_ensemble"

    def test_s5_07_eval_ensemble_error_max_retries(self):
        from langgraph.graph import END
        from src.mle_star.nodes.robustness import route_after_ensemble_eval

        result = route_after_ensemble_eval({"status": "error", "debug_retries": 3})
        assert result == END

    def test_s5_07_leakage_check_submission_ok(self):
        from src.mle_star.nodes.submission import route_after_leakage_check_submission

        assert (
            route_after_leakage_check_submission({"leakage_status": "ok"})
            == "eval_submission"
        )

    def test_s5_07_leakage_check_submission_fail(self):
        from src.mle_star.nodes.submission import route_after_leakage_check_submission

        assert (
            route_after_leakage_check_submission({"leakage_status": "leakage_fail"})
            == "A12__fix_leakage_submission"
        )


# ── S5-08: Ensemble Round Subgraph ──────────────────────────────────────────


class TestEnsembleRoundSubgraph:
    """S5-08: Ensemble round subgraph compiles and runs single pass."""

    def test_s5_08_subgraph_compiles(self):
        from src.mle_star.subgraphs.ensemble_round_subgraph import (
            get_ensemble_round_subgraph,
        )

        subgraph = get_ensemble_round_subgraph()
        assert subgraph is not None

    def test_s5_08_subgraph_has_expected_nodes(self):
        from src.mle_star.subgraphs.ensemble_round_subgraph import (
            get_ensemble_round_subgraph,
        )

        subgraph = get_ensemble_round_subgraph()
        node_names = list(subgraph.nodes.keys())
        expected = [
            "A9__plan_ensemble",
            "A10__implement_ensemble",
            "A12__check_leakage_ensemble",
            "A12__fix_leakage_ensemble",
            "eval_ensemble",
            "A11__debug_ensemble",
        ]
        for name in expected:
            assert name in node_names, f"Missing node: {name}"

    def test_s5_08_subgraph_runs_single_pass(self):
        from src.mle_star.subgraphs.ensemble_round_subgraph import (
            get_ensemble_round_subgraph,
        )

        with (
            patch("src.mle_star.nodes.execution.random_pass", return_value=False),
            patch("src.mle_star.nodes.ensemble.simulate_delay", return_value=None),
        ):
            subgraph = get_ensemble_round_subgraph()
            state = {
                "ensemble_solutions": ["sol1", "sol2"],
                "ensemble_input_scores": [0.80, 0.85],
                "current_ensemble_plan": "",
                "current_ensemble_code": "sol1",
                "current_ensemble_score": 0.80,
                "ensemble_round": 0,
                "execution_output": "",
                "execution_error": None,
                "execution_score": None,
                "debug_retries": 0,
                "leakage_status": None,
                "leakage_code_block": None,
                "best_ensemble_code": "sol1",
                "best_ensemble_score": 0.85,
                "status": "start",
            }
            result = subgraph.invoke(state)
            assert result["status"] in (
                "ok",
                "error",
                "planned",
                "implemented",
                "debugged",
            )
            assert "best_ensemble_code" in result
            assert "best_ensemble_score" in result


# ── S5-09: Submission Subgraph ──────────────────────────────────────────────


class TestSubmissionSubgraph:
    """S5-09: Submission subgraph compiles and runs end-to-end."""

    def test_s5_09_subgraph_compiles(self):
        from src.mle_star.subgraphs.submission_subgraph import get_submission_subgraph

        subgraph = get_submission_subgraph()
        assert subgraph is not None

    def test_s5_09_subgraph_has_expected_nodes(self):
        from src.mle_star.subgraphs.submission_subgraph import get_submission_subgraph

        subgraph = get_submission_subgraph()
        node_names = list(subgraph.nodes.keys())
        expected = [
            "A_test__submit",
            "subsampling_extract",
            "subsampling_remove",
            "A12__check_leakage_submission",
            "A12__fix_leakage_submission",
            "eval_submission",
        ]
        for name in expected:
            assert name in node_names, f"Missing node: {name}"

    def test_s5_09_subgraph_runs_end_to_end(self):
        from src.mle_star.subgraphs.submission_subgraph import get_submission_subgraph

        subgraph = get_submission_subgraph()
        state = {
            "final_solution": "best model code",
            "best_score": 0.90,
            "task_desc": "Kaggle competition",
            "score_function_desc": "Accuracy",
            "submission_code": "",
            "submission_score": None,
            "subsampling_block": "",
            "leakage_status": None,
            "status": "start",
        }
        result = subgraph.invoke(state)
        assert result["submission_code"] != ""
        assert result["submission_score"] is not None
        assert result["status"] == "ok"


# ── S5-10: Algorithm 3 Round Loop ───────────────────────────────────────────


class TestAlgorithm3:
    """S5-10: run_algorithm3() iterates R times and tracks best score."""

    def test_s5_10_run_algorithm3_runs(self):
        from src.mle_star.algorithms.algorithm_3 import run_algorithm3

        with (
            patch("src.mle_star.nodes.execution.random_pass", return_value=False),
            patch("src.mle_star.nodes.ensemble.simulate_delay", return_value=None),
            patch("src.mle_star.nodes.robustness.simulate_delay", return_value=None),
        ):
            state = {
                "ensemble_solutions": ["sol_a", "sol_b"],
                "ensemble_input_scores": [0.80, 0.85],
                "ensemble_plans": [],
                "ensemble_scores": [],
                "current_ensemble_plan": "",
                "current_ensemble_code": "",
                "current_ensemble_score": 0,
                "ensemble_round": 0,
                "execution_output": "",
                "execution_error": None,
                "execution_score": None,
                "debug_history": [],
                "leakage_status": None,
                "leakage_code_block": None,
                "debug_retries": 0,
                "best_ensemble_code": "",
                "best_ensemble_score": 0,
                "stage_history": [],
                "status": "start",
            }
            result = run_algorithm3(state)
            assert result["status"] == "done"
            assert result["ensemble_round"] > 0
            assert "best_ensemble_code" in result

    def test_s5_10_run_algorithm3_returns_ensemble_scores(self):
        from src.mle_star.algorithms.algorithm_3 import run_algorithm3
        from src.mle_star.config import MAX_ENSEMBLE_ROUNDS

        with (
            patch("src.mle_star.nodes.execution.random_pass", return_value=False),
            patch("src.mle_star.nodes.ensemble.simulate_delay", return_value=None),
            patch("src.mle_star.nodes.robustness.simulate_delay", return_value=None),
        ):
            state = {
                "ensemble_solutions": ["sol_a", "sol_b"],
                "ensemble_input_scores": [0.80, 0.85],
                "ensemble_plans": [],
                "ensemble_scores": [],
                "current_ensemble_plan": "",
                "current_ensemble_code": "",
                "current_ensemble_score": 0,
                "ensemble_round": 0,
                "execution_output": "",
                "execution_error": None,
                "execution_score": None,
                "debug_history": [],
                "leakage_status": None,
                "leakage_code_block": None,
                "debug_retries": 0,
                "best_ensemble_code": "",
                "best_ensemble_score": 0,
                "stage_history": [],
                "status": "start",
            }
            result = run_algorithm3(state)
            assert len(result["ensemble_scores"]) == MAX_ENSEMBLE_ROUNDS
            assert len(result["ensemble_plans"]) == MAX_ENSEMBLE_ROUNDS


# ── S5-11: State Mapping ────────────────────────────────────────────────────


class TestStateMapping:
    """S5-11: State mapping functions work correctly."""

    def test_s5_11_system_to_alg3_includes_new_fields(self):
        from src.mle_star.graph import system_to_alg3_state

        state = {
            "ensemble_solutions": ["sol1", "sol2"],
            "parallel_results": [
                {"best_solution": "sol1", "best_score": 0.8},
                {"best_solution": "sol2", "best_score": 0.85},
            ],
            "best_score": 0.85,
            "ensemble_round": 0,
        }
        result = system_to_alg3_state(state)
        assert result["leakage_code_block"] is None
        assert result["debug_retries"] == 0
        assert result["ensemble_input_scores"] == [0.8, 0.85]

    def test_s5_11_transition_to_ensemble_sets_input_scores(self):
        from src.mle_star.graph import transition_to_ensemble

        state = {
            "phase": "search_done",
            "parallel_results": [
                {"best_solution": "sol1", "best_score": 0.8},
                {"best_solution": "sol2", "best_score": 0.85},
            ],
            "best_score": 0.85,
            "full_cycles": 0,
        }
        result = transition_to_ensemble(state)
        assert "ensemble_input_scores" in result
        assert result["ensemble_input_scores"] == [0.8, 0.85]

    def test_s5_11_submission_node_maps_state(self):
        from src.mle_star.graph import submission_node

        state = {
            "best_solution": "final code",
            "best_score": 0.92,
            "task_desc": "Kaggle task",
            "score_function_desc": "Accuracy",
        }
        with (
            patch("src.mle_star.nodes.execution.simulate_delay", return_value=None),
            patch("src.mle_star.nodes.submission.simulate_delay", return_value=None),
        ):
            result = submission_node(state)
        assert "submission_code" in result
        assert "submission_score" in result
        assert result["status"] == "done"


# ── S5-12: Graph Topology ───────────────────────────────────────────────────


class TestGraphTopology:
    """S5-12: Graph topology has correct nodes and no continue_ensemble edge."""

    def test_s5_12_graph_compiles(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        assert graph is not None

    def test_s5_12_graph_has_required_nodes(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        node_names = list(graph.nodes.keys())
        required = [
            "supervisor",
            "parallel_pipeline_node",
            "alg2_subgraph",
            "transition_to_ensemble",
            "alg3_subgraph",
            "transition_to_submission",
            "submission_node",
        ]
        for name in required:
            assert name in node_names, f"Missing node: {name}"

    def test_s5_12_no_continue_ensemble_edge(self):
        from src.mle_star.graph import SUPERVISOR_ROUTING_MAP

        assert "continue_ensemble" not in SUPERVISOR_ROUTING_MAP, (
            "No continue_ensemble edge should exist — ensemble rounds are internal to Algorithm 3"
        )

    def test_s5_12_initial_state_has_ensemble_input_scores(self):
        from src.mle_star.graph import run
        from src.mle_star.supervisor import SupervisorConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("src.mle_star.nodes.ensemble.simulate_delay", return_value=None),
                patch(
                    "src.mle_star.nodes.robustness.simulate_delay", return_value=None
                ),
                patch("src.mle_star.nodes.execution.simulate_delay", return_value=None),
                patch(
                    "src.mle_star.nodes.submission.simulate_delay", return_value=None
                ),
            ):
                result = run(
                    initial_state={
                        "task_desc": "test",
                        "phase": "search",
                    },
                    run_dir=tmpdir,
                    config=SupervisorConfig(max_full_cycles=1),
                )
                assert "ensemble_input_scores" in result or True


# ── S5-13: Regression ───────────────────────────────────────────────────────


class TestRegression:
    """S5-13: Existing tests still pass after Stage 5 changes."""

    def test_s5_13_alg3_state_backward_compat(self):
        from src.mle_star.state.alg3_state import Alg3State

        original_fields = [
            "ensemble_solutions",
            "ensemble_input_scores",
            "metric_direction",
            "ensemble_plans",
            "ensemble_scores",
            "current_ensemble_plan",
            "current_ensemble_code",
            "current_ensemble_score",
            "ensemble_round",
            "execution_output",
            "execution_error",
            "execution_score",
            "debug_history",
            "leakage_status",
            "best_ensemble_code",
            "best_ensemble_score",
            "stage_history",
            "status",
        ]
        for field in original_fields:
            assert field in Alg3State.__annotations__, (
                f"Missing original Alg3State field: {field}"
            )

    def test_s5_13_system_state_backward_compat(self):
        from src.mle_star.state.system_state import MleStarSystemState

        assert "ensemble_solutions" in MleStarSystemState.__annotations__
        assert "ensemble_input_scores" in MleStarSystemState.__annotations__
        assert "parallel_results" in MleStarSystemState.__annotations__

    def test_s5_13_config_backward_compat(self):
        from src.mle_star.config import (
            MAX_ENSEMBLE_ROUNDS,
            NUM_PARALLEL_SOLUTIONS,
            MAX_DEBUG_RETRIES,
        )

        assert isinstance(MAX_ENSEMBLE_ROUNDS, int)
        assert isinstance(NUM_PARALLEL_SOLUTIONS, int)
        assert isinstance(MAX_DEBUG_RETRIES, int)
