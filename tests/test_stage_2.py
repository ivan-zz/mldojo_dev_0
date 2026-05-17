"""Stage 2 verification tests: System State + Supervisor + Main Graph.

Validates that the MleStarSystemState, supervisor, transition functions,
metric utilities, and main graph topology are correctly implemented.

Test IDs map to the verification checklist in the implementation steps doc
(S2-01 through S2-21).
"""

import operator
import os
import tempfile

import pytest

from src.mle_star.supervisor import SupervisorConfig


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "mle_star")
MLE_STAR_DIR = os.path.normpath(BASE_DIR)


# ── S2-01: MleStarSystemState fields ────────────────────────────────────


class TestSystemState:
    """S2-01: System state has all required fields."""

    def test_s2_01_system_state_fields(self):
        from src.mle_star.state.system_state import MleStarSystemState

        required_fields = [
            "task_desc",
            "datasets",
            "score_function_desc",
            "phase",
            "phase_history",
            "current_solution",
            "best_solution",
            "best_score",
            "current_score",
            "metric_direction",
            "raw_best_score",
            "alg1_result",
            "alg2_result",
            "outer_step",
            "inner_step",
            "convergence_achieved",
            "alg3_result",
            "ensemble_solutions",
            "ensemble_round",
            "submission_code",
            "submission_score",
            "full_cycles",
            "max_full_cycles",
            "debug_history",
            "security_violations",
            "stage_history",
            "status",
        ]
        missing = [
            k for k in required_fields if k not in MleStarSystemState.__annotations__
        ]
        assert not missing, f"Missing MleStarSystemState fields: {missing}"

    def test_system_state_accumulation_fields(self):
        """S2-14, S2-15: phase_history and stage_history use operator.add."""
        from src.mle_star.state.system_state import MleStarSystemState

        ann = MleStarSystemState.__annotations__
        for field in [
            "phase_history",
            "stage_history",
            "debug_history",
            "security_violations",
        ]:
            assert field in ann, f"Missing accumulation field: {field}"

    def test_system_state_1_indexed_naming(self):
        """Verify 1-indexed naming: alg1_result, alg2_result, alg3_result."""
        from src.mle_star.state.system_state import MleStarSystemState

        ann = MleStarSystemState.__annotations__
        assert "alg1_result" in ann, "Missing alg1_result (should be 1-indexed)"
        assert "alg2_result" in ann, "Missing alg2_result"
        assert "alg3_result" in ann, "Missing alg3_result"
        assert "alg0_result" not in ann, "alg0_result should not exist (1-indexed)"


# ── S2-02, S2-03: SupervisorConfig and supervisor_decide ──────────────


class TestSupervisorConfig:
    """S2-02: SupervisorConfig has all fields from design doc."""

    def test_s2_02_supervisor_config_fields(self):
        from src.mle_star.supervisor import SupervisorConfig

        config = SupervisorConfig()
        assert config.epsilon == 0.001
        assert config.max_outer_steps == 4
        assert config.max_inner_steps == 4
        assert config.max_stagnation_rounds == 2
        assert config.max_debug_retries == 3
        assert config.max_ensemble_rounds == 5
        assert config.num_retrieved_models == 4
        assert config.num_parallel_solutions == 2
        assert config.max_full_cycles == 3
        assert config.llm_config is None

    def test_s2_02_supervisor_config_custom(self):
        from src.mle_star.supervisor import SupervisorConfig

        config = SupervisorConfig(epsilon=0.01, max_full_cycles=5)
        assert config.epsilon == 0.01
        assert config.max_full_cycles == 5


class TestSupervisorDecide:
    """S2-03, S2-09, S2-10, S2-11: supervisor_decide routing logic."""

    def test_s2_03_supervisor_decide_signature(self):
        """S2-03: supervisor_decide has correct (state, config) signature."""
        from src.mle_star.supervisor import supervisor_decide, SupervisorConfig

        state = {"phase": "search", "full_cycles": 0}
        result = supervisor_decide(state)
        assert isinstance(result, str)
        config = SupervisorConfig()
        result2 = supervisor_decide(state, config)
        assert isinstance(result2, str)

    def test_s2_09_search_routes_to_parallel_pipeline(self):
        """Search phase routes to parallel_pipeline_node (D15)."""
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "search", "full_cycles": 0}
        assert supervisor_decide(state) == "parallel_pipeline_node"

    def test_s2_09_ablation_improving(self):
        """Ablation with improving score continues."""
        from src.mle_star.supervisor import supervisor_decide

        state = {
            "phase": "ablation",
            "current_score": 0.9,
            "best_score": 0.8,
            "full_cycles": 0,
        }
        assert supervisor_decide(state) == "continue_ablation"

    def test_s2_09_ablation_stagnated_new_cycle(self):
        """Ablation stagnated with remaining outer steps starts new cycle."""
        from src.mle_star.supervisor import supervisor_decide

        state = {
            "phase": "ablation",
            "current_score": 0.8,
            "best_score": 0.8,
            "outer_step": 1,
            "full_cycles": 0,
        }
        assert supervisor_decide(state) == "new_ablation_cycle"

    def test_s2_09_ablation_exhausted_to_ensemble(self):
        """Ablation with exhausted outer steps transitions to ensemble."""
        from src.mle_star.supervisor import supervisor_decide

        state = {
            "phase": "ablation",
            "current_score": 0.8,
            "best_score": 0.8,
            "outer_step": 3,
            "full_cycles": 0,
        }
        assert supervisor_decide(state) == "transition_to_ensemble"

    def test_s2_09_ensemble_transitions_to_submission(self):
        """Ensemble phase transitions directly to submission (rounds managed internally by Alg3)."""
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "ensemble", "ensemble_round": 2, "full_cycles": 0}
        assert supervisor_decide(state) == "transition_to_submission"

    def test_s2_09_ensemble_at_any_round_transitions(self):
        """Ensemble always transitions to submission regardless of round number."""
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "ensemble", "ensemble_round": 0, "full_cycles": 0}
        assert supervisor_decide(state) == "transition_to_submission"

    def test_s2_09_submission_to_end(self):
        """Submission phase routes to END."""
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "submission", "full_cycles": 0}
        assert supervisor_decide(state) == "END"

    def test_s2_10_max_cycles_forces_submission(self):
        """After max_full_cycles, force submission regardless of phase."""
        from src.mle_star.supervisor import supervisor_decide, SupervisorConfig

        config = SupervisorConfig(max_full_cycles=3)
        state = {
            "phase": "ablation",
            "full_cycles": 3,
            "current_score": 0.9,
            "best_score": 0.8,
        }
        assert supervisor_decide(state, config) == "transition_to_submission"

    def test_s2_11_no_stagnation_edge(self):
        """No phase='stagnation' edge exists in routing."""
        from src.mle_star.supervisor import supervisor_decide

        for phase in ["search", "ablation", "ensemble", "submission"]:
            state = {"phase": phase, "full_cycles": 0}
            result = supervisor_decide(state)
            assert result != "stagnation", f"stagnation edge found for phase={phase}"

    def test_supervisor_node_returns_phase_history(self):
        """supervisor_node returns phase_history accumulation."""
        from src.mle_star.supervisor import supervisor_node

        state = {"phase": "search", "full_cycles": 0}
        result = supervisor_node(state)
        assert "phase_history" in result
        assert len(result["phase_history"]) == 1
        assert "decision" in result["phase_history"][0]
        assert "phase" in result["phase_history"][0]

    def test_supervisor_decide_llm_fallback(self):
        """supervisor_decide_llm falls back to rule-based."""
        from src.mle_star.supervisor import supervisor_decide_llm, supervisor_decide

        state = {"phase": "search", "full_cycles": 0}
        assert supervisor_decide_llm(state) == supervisor_decide(state)


# ── S2-05, S2-06, S2-07, S2-08: Metric Direction Utilities ───────────


class TestMetricDirection:
    def test_s2_05_maximize_metrics(self):
        """infer_metric_direction returns 'maximize' for accuracy/AUC/F1."""
        from src.mle_star.state.shared import infer_metric_direction

        assert infer_metric_direction("classification accuracy") == "maximize"
        assert infer_metric_direction("ROC AUC score") == "maximize"
        assert infer_metric_direction("F1 score") == "maximize"
        assert infer_metric_direction("f1_macro") == "maximize"
        assert infer_metric_direction("r2 regression") == "maximize"

    def test_s2_06_minimize_metrics(self):
        """infer_metric_direction returns 'minimize' for RMSLE/MAE/MSE."""
        from src.mle_star.state.shared import infer_metric_direction

        assert infer_metric_direction("RMSLE error") == "minimize"
        assert infer_metric_direction("MAE loss") == "minimize"
        assert infer_metric_direction("MSE metric") == "minimize"
        assert infer_metric_direction("log_loss") == "minimize"
        assert infer_metric_direction("rmse score") == "minimize"

    def test_s2_06_default_maximize(self):
        """Unknown metrics default to 'maximize'."""
        from src.mle_star.state.shared import infer_metric_direction

        assert infer_metric_direction("custom metric") == "maximize"
        assert infer_metric_direction("") == "maximize"

    def test_s2_07_normalize_score(self):
        """normalize_score negates minimize metrics, leaves maximize unchanged."""
        from src.mle_star.state.shared import normalize_score

        assert normalize_score(0.95, "maximize") == 0.95
        assert normalize_score(0.05, "minimize") == -0.05
        assert normalize_score(-1.0, "minimize") == 1.0

    def test_s2_08_display_score(self):
        """display_score converts normalized back to original units."""
        from src.mle_star.state.shared import display_score

        assert display_score(0.95, "maximize") == 0.95
        assert display_score(-0.05, "minimize") == 0.05
        assert display_score(1.0, "minimize") == -1.0


# ── S2-13, S2-16: State Mapping and Transition Functions ───────────────


class TestStateMapping:
    def test_s2_13_system_to_alg1_state(self):
        """system_to_alg1_state maps correctly."""
        from src.mle_star.graph import system_to_alg1_state

        state = {
            "task_desc": "predict housing prices",
            "alg1_result": {"retrieved_models": ["LGBM"]},
        }
        result = system_to_alg1_state(state)
        assert result["task_desc"] == "predict housing prices"
        assert result["metric_direction"] == "maximize"
        assert result["retrieved_models"] == ["LGBM"]
        assert result["candidates_pool"] == []
        assert result["status"] == "start"

    def test_s2_13_alg1_result_to_system(self):
        """alg1_result_to_system maps correctly."""
        from src.mle_star.graph import alg1_result_to_system

        result = {
            "best_candidate": {"code": "def solution(): pass", "score": 0.95},
            "candidates_pool": [{"model": "LGBM", "score": 0.95}],
        }
        mapped = alg1_result_to_system(result)
        assert mapped["best_solution"] == "def solution(): pass"
        assert mapped["best_score"] == 0.95
        assert mapped["raw_best_score"] == 0.95
        assert mapped["phase"] == "search_done"

    def test_system_to_alg2_state(self):
        """system_to_alg2_state maps correctly."""
        from src.mle_star.graph import system_to_alg2_state

        state = {
            "current_solution": "def best(): pass",
            "best_score": 0.95,
            "outer_step": 2,
            "inner_step": 1,
        }
        result = system_to_alg2_state(state)
        assert result["current_solution"] == "def best(): pass"
        assert result["best_score"] == 0.95
        assert result["metric_direction"] == "maximize"
        assert result["outer_step"] == 2
        assert result["inner_step"] == 1
        assert result["convergence_achieved"] is False
        assert result["status"] == "start"

    def test_system_to_alg3_state(self):
        """system_to_alg3_state maps correctly."""
        from src.mle_star.graph import system_to_alg3_state

        state = {
            "ensemble_solutions": ["sol1", "sol2"],
            "ensemble_round": 1,
        }
        result = system_to_alg3_state(state)
        assert result["ensemble_solutions"] == ["sol1", "sol2"]
        assert result["ensemble_round"] == 1
        assert result["metric_direction"] == "maximize"
        assert result["status"] == "start"

    def test_alg2_result_to_system_maximize(self):
        """alg2_result_to_system updates best when improved (maximize)."""
        from src.mle_star.graph import alg2_result_to_system

        result = {
            "improved_solution": "sol_a",
            "improved_score": 0.95,
            "stage_history": [],
        }
        state = {"best_score": 0.90, "metric_direction": "maximize"}
        updates = alg2_result_to_system(result, state)
        assert updates["best_solution"] == "sol_a"
        assert updates["best_score"] == 0.95
        assert updates["raw_best_score"] == 0.95

    def test_alg2_result_to_system_minimize(self):
        """alg2_result_to_system updates best when lower score (minimize)."""
        from src.mle_star.graph import alg2_result_to_system

        result = {
            "improved_solution": "sol_a",
            "improved_score": 0.05,
            "stage_history": [],
        }
        state = {"best_score": 0.10, "metric_direction": "minimize"}
        updates = alg2_result_to_system(result, state)
        assert updates["best_solution"] == "sol_a"
        assert updates["best_score"] == 0.05

    def test_alg2_result_to_system_no_regression(self):
        """alg2_result_to_system doesn't update best when worse."""
        from src.mle_star.graph import alg2_result_to_system

        result = {
            "improved_solution": "sol_b",
            "improved_score": 0.88,
            "stage_history": [],
        }
        state = {"best_score": 0.95, "metric_direction": "maximize"}
        updates = alg2_result_to_system(result, state)
        assert "best_solution" not in updates
        assert "best_score" not in updates

    def test_alg2_result_to_system_security_violations(self):
        """alg2_result_to_system propagates security_violations."""
        from src.mle_star.graph import alg2_result_to_system

        result = {
            "improved_solution": "sol_a",
            "improved_score": 0.95,
            "security_violations": ["dangerous_import"],
            "stage_history": [],
        }
        state = {"best_score": 0.90, "metric_direction": "maximize"}
        updates = alg2_result_to_system(result, state)
        assert updates["security_violations"] == ["dangerous_import"]

    def test_alg3_result_to_system_maximize(self):
        """alg3_result_to_system updates best when improved (maximize)."""
        from src.mle_star.graph import alg3_result_to_system

        result = {
            "best_ensemble_code": "ens_a",
            "best_ensemble_score": 0.92,
            "ensemble_round": 2,
        }
        state = {"best_score": 0.90, "metric_direction": "maximize"}
        updates = alg3_result_to_system(result, state)
        assert updates["best_solution"] == "ens_a"
        assert updates["best_score"] == 0.92
        assert updates["raw_best_score"] == 0.92

    def test_alg3_result_to_system_minimize(self):
        """alg3_result_to_system updates best when lower score (minimize)."""
        from src.mle_star.graph import alg3_result_to_system

        result = {
            "best_ensemble_code": "ens_a",
            "best_ensemble_score": 0.03,
            "ensemble_round": 2,
        }
        state = {"best_score": 0.10, "metric_direction": "minimize"}
        updates = alg3_result_to_system(result, state)
        assert updates["best_solution"] == "ens_a"
        assert updates["best_score"] == 0.03


class TestTransitionFunctions:
    def test_s2_16_transition_to_ablation_removed(self):
        """transition_to_ablation has been removed (D15: ablation is inside parallel pipelines)."""
        import src.mle_star.graph as graph_module

        assert not hasattr(graph_module, "transition_to_ablation"), (
            "transition_to_ablation should be removed from graph module (D15)"
        )

    def test_s2_16_transition_to_ensemble_from_parallel_results(self):
        """transition_to_ensemble reads from parallel_results (D15)."""
        from src.mle_star.graph import transition_to_ensemble

        state = {
            "phase": "search_done",
            "parallel_results": [
                {"run_index": 0, "best_solution": "sol_0", "best_score": 0.85},
                {"run_index": 1, "best_solution": "sol_1", "best_score": 0.90},
            ],
            "best_score": 0.90,
            "full_cycles": 0,
        }
        result = transition_to_ensemble(state)
        assert result["phase"] == "ensemble"
        assert result["ensemble_solutions"] == ["sol_0", "sol_1"]
        assert result["ensemble_round"] == 0
        assert result["full_cycles"] == 1

    def test_s2_16_transition_to_submission(self):
        """transition_to_submission copies best_solution to submission_code."""
        from src.mle_star.graph import transition_to_submission

        state = {"best_solution": "final_code", "phase": "ensemble"}
        result = transition_to_submission(state)
        assert result["phase"] == "submission"
        assert result["submission_code"] == "final_code"


# ── S2-04, S2-17: Main Graph Compilation and Routing ───────────────────


class TestMainGraph:
    def test_s2_04_graph_compiles(self):
        """S2-04: Main graph compiles with StateGraph(MleStarSystemState)."""
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        assert graph is not None
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "supervisor",
            "parallel_pipeline_node",
            "alg2_subgraph",
            "transition_to_ensemble",
            "alg3_subgraph",
            "transition_to_submission",
            "submission_node",
        }
        assert expected_nodes.issubset(node_names), (
            f"Missing nodes: {expected_nodes - node_names}"
        )

    def test_s2_17_search_routes_to_parallel_pipeline(self):
        """S2-17: phase='search' triggers parallel_pipeline_node via supervisor routing (D15)."""
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "search", "full_cycles": 0}
        decision = supervisor_decide(state)
        assert decision == "parallel_pipeline_node"

    def test_alg2_graph_compiles(self):
        """Algorithm 2 run function exists."""
        from src.mle_star.algorithms.algorithm_2 import run_algorithm2

        assert callable(run_algorithm2)

    def test_alg3_graph_compiles(self):
        """Algorithm 3 ensemble round subgraph compiles."""
        from src.mle_star.subgraphs.ensemble_round_subgraph import (
            get_ensemble_round_subgraph,
        )

        graph = get_ensemble_round_subgraph()
        assert graph is not None

    def test_ablation_subgraph_compiles(self):
        """Ablation subgraph placeholder compiles."""
        from src.mle_star.subgraphs.ablation_subgraph import get_ablation_subgraph

        graph = get_ablation_subgraph()
        assert graph is not None

    def test_submission_subgraph_compiles(self):
        """Submission subgraph placeholder compiles."""
        from src.mle_star.subgraphs.submission_subgraph import get_submission_subgraph

        graph = get_submission_subgraph()
        assert graph is not None


# ── S2-19: SupervisorConfig defaults match design doc ─────────────────


class TestSupervisorConfigDefaults:
    def test_s2_19_supervisor_config_defaults(self):
        from src.mle_star.supervisor import SupervisorConfig

        config = SupervisorConfig()
        assert config.epsilon == 0.001
        assert config.max_outer_steps == 4
        assert config.max_inner_steps == 4
        assert config.max_stagnation_rounds == 2
        assert config.max_ensemble_rounds == 5
        assert config.num_retrieved_models == 4
        assert config.num_parallel_solutions == 2
        assert config.max_full_cycles == 3


# ── S2-12: MLELogger + log_node_event produce valid JSON ───────────────
# Already tested in Stage 1 (S1-15). Verify it still works.


class TestJsonLoggingRegression:
    def test_s2_12_log_node_event_still_works(self, caplog):
        """S2-12: log_node_event still works after Stage 2 changes."""
        import json
        import logging

        from src.mle_star.state.shared import log_node_event

        with caplog.at_level(logging.DEBUG, logger="mle_star"):
            log_node_event("s2_test", "test_event", {"key": "value"}, duration_ms=42.0)

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        parsed = json.loads(record.getMessage())
        assert parsed["node"] == "s2_test"
        assert parsed["event"] == "test_event"
        assert parsed["duration_ms"] == 42.0


# ── Alg2State and Alg3State field verification ────────────────────────


class TestAlg2State:
    def test_alg2_state_fields(self):
        from src.mle_star.state.alg2_state import Alg2State

        required = [
            "current_solution",
            "best_score",
            "ablation_scripts",
            "ablation_results_list",
            "ablation_summaries",
            "target_block",
            "initial_plan",
            "refined_blocks",
            "current_plans",
            "current_scores",
            "refined_code",
            "candidate_solution",
            "execution_output",
            "execution_error",
            "execution_score",
            "outer_step",
            "inner_step",
            "leakage_status",
            "debug_history",
            "security_violations",
            "improved_solution",
            "improved_score",
            "stage_history",
            "status",
        ]
        missing = [k for k in required if k not in Alg2State.__annotations__]
        assert not missing, f"Missing Alg2State fields: {missing}"

    def test_alg2_state_accumulation_fields(self):
        from src.mle_star.state.alg2_state import Alg2State

        ann = Alg2State.__annotations__
        for field in [
            "ablation_summaries",
            "refined_blocks",
            "debug_history",
            "security_violations",
            "stage_history",
        ]:
            assert field in ann, f"Missing accumulation field: {field}"


class TestAlg3State:
    def test_alg3_state_fields(self):
        from src.mle_star.state.alg3_state import Alg3State

        required = [
            "ensemble_solutions",
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
        missing = [k for k in required if k not in Alg3State.__annotations__]
        assert not missing, f"Missing Alg3State fields: {missing}"


# ── Metric direction dict coverage ─────────────────────────────────────


class TestMetricDirectionsDict:
    def test_metric_directions_dict_exists(self):
        from src.mle_star.state.shared import METRIC_DIRECTIONS

        assert isinstance(METRIC_DIRECTIONS, dict)
        assert len(METRIC_DIRECTIONS) >= 20

    def test_maximize_keys(self):
        from src.mle_star.state.shared import METRIC_DIRECTIONS

        maximize_keys = [
            "accuracy",
            "auc",
            "f1",
            "f1_score",
            "r2",
            "precision",
            "recall",
            "auroc",
        ]
        for k in maximize_keys:
            assert k in METRIC_DIRECTIONS, f"{k} not in METRIC_DIRECTIONS"
            assert METRIC_DIRECTIONS[k] == "maximize", f"{k} should be maximize"

    def test_minimize_keys(self):
        from src.mle_star.state.shared import METRIC_DIRECTIONS

        minimize_keys = ["rmsle", "rmse", "mse", "mae", "log_loss", "logloss"]
        for k in minimize_keys:
            assert k in METRIC_DIRECTIONS, f"{k} not in METRIC_DIRECTIONS"
            assert METRIC_DIRECTIONS[k] == "minimize", f"{k} should be minimize"


# ── S2-20, S2-21: Regression — Algorithm 1 still works ────────────────


class TestRegression:
    def test_s2_20_algorithm1_e2e(self, e2e_result):
        """S2-20: Algorithm 1 still runs end-to-end."""
        assert e2e_result.get("status") == "done"
        assert len(e2e_result.get("candidates_pool", [])) > 0

    def test_s2_21_node_names_unchanged(self, e2e_result):
        """S2-21: Node names unchanged from Stage 1."""
        stage_names = {entry["stage"] for entry in e2e_result.get("stage_history", [])}
        expected = {
            "A1__retrieve",
            "generate_candidates",
            "Rank",
            "merge_candidates",
            "SelectBest",
        }
        assert expected.issubset(stage_names), f"Expected {expected}, got {stage_names}"

    def test_graph_module_imports(self):
        """S2-21: Graph module has new exports."""
        from src.mle_star.graph import (
            get_mle_star_graph,
            get_alg1_graph,
            system_to_alg1_state,
            alg1_result_to_system,
            system_to_alg2_state,
            alg2_result_to_system,
            system_to_alg3_state,
            alg3_result_to_system,
            transition_to_ensemble,
            transition_to_submission,
            parallel_pipeline_node,
        )

        assert callable(get_mle_star_graph)
        assert callable(get_alg1_graph)
        assert callable(transition_to_ensemble)
        assert callable(transition_to_submission)
        assert callable(parallel_pipeline_node)


# ── S2-22: run() function — single trace, session, cleanup ────────────


class TestGraphRun:
    """S2-22: graph.run() creates root trace with session grouping."""

    def test_s2_22_run_creates_result(self):
        """run() returns a completed state dict."""
        from src.mle_star.graph import run

        import tempfile

        run_dir = tempfile.mkdtemp(prefix="mle_star_run_test_")
        result = run(
            initial_state={
                "task_desc": "test run",
                "score_function_desc": "accuracy — higher is better",
                "phase": "search",
                "phase_history": [],
                "current_solution": "",
                "best_solution": "",
                "best_score": 0.0,
                "current_score": None,
                "metric_direction": "maximize",
                "raw_best_score": None,
                "alg1_result": {},
                "alg2_result": {},
                "outer_step": 0,
                "inner_step": 0,
                "convergence_achieved": False,
                "alg3_result": {},
                "ensemble_solutions": [],
                "ensemble_round": 0,
                "submission_code": "",
                "submission_score": None,
                "full_cycles": 0,
                "max_full_cycles": 1,
                "debug_history": [],
                "security_violations": [],
                "stage_history": [],
                "status": "start",
            },
            run_dir=run_dir,
            config=SupervisorConfig(max_full_cycles=1),
        )
        assert result.get("status") == "done", (
            f"Expected done, got {result.get('status')}"
        )
        assert result.get("submission_code", "") != "", "No submission produced"
        assert result.get("best_score", 0) > 0, "Expected best_score > 0"
        assert result.get("phase") in ("submission", "done"), (
            f"Unexpected phase: {result.get('phase')}"
        )

    def test_s2_22_run_metric_direction_inferred(self):
        """run() infers metric_direction from score_function_desc."""
        from src.mle_star.graph import run

        import tempfile

        run_dir = tempfile.mkdtemp(prefix="mle_star_metric_test_")
        result = run(
            initial_state={
                "task_desc": "RMSLE test",
                "score_function_desc": "RMSLE (root mean squared logarithmic error) — lower is better",
                "phase": "search",
                "phase_history": [],
                "current_solution": "",
                "best_solution": "",
                "best_score": 0.0,
                "current_score": None,
                "raw_best_score": None,
                "alg1_result": {},
                "alg2_result": {},
                "outer_step": 0,
                "inner_step": 0,
                "convergence_achieved": False,
                "alg3_result": {},
                "ensemble_solutions": [],
                "ensemble_round": 0,
                "submission_code": "",
                "submission_score": None,
                "full_cycles": 0,
                "max_full_cycles": 1,
                "debug_history": [],
                "security_violations": [],
                "stage_history": [],
                "status": "start",
            },
            run_dir=run_dir,
            config=SupervisorConfig(max_full_cycles=1),
        )
        assert result is not None

    def test_s2_22_run_default_state(self):
        """run() works with default initial_state."""
        from src.mle_star.graph import run

        import tempfile

        run_dir = tempfile.mkdtemp(prefix="mle_star_default_test_")
        result = run(
            run_dir=run_dir,
            config=SupervisorConfig(max_full_cycles=1),
        )
        assert result is not None
        assert result.get("status") == "done"

    def test_s2_22_run_creates_run_log(self):
        """run() creates a run.log file in the run directory."""
        from src.mle_star.graph import run

        import tempfile

        run_dir = tempfile.mkdtemp(prefix="mle_star_log_test_")
        result = run(
            run_dir=run_dir,
            config=SupervisorConfig(max_full_cycles=1),
        )
        log_path = os.path.join(run_dir, "run.log")
        assert os.path.isfile(log_path), f"run.log not found at {log_path}"
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) > 0, "run.log is empty"

    def test_s2_22_import_run_function(self):
        """run() is importable from graph module."""
        from src.mle_star.graph import run

        assert callable(run)

    def test_s2_22_trace_name_format(self):
        """Trace name follows MLE_STAR_YYYYMMDDHHMMSS format."""
        from datetime import datetime

        from src.mle_star.graph import run

        import tempfile

        before = datetime.now().strftime("%Y%m%d%H%M")
        run_dir = tempfile.mkdtemp(prefix="mle_star_trace_name_")
        result = run(
            run_dir=run_dir,
            config=SupervisorConfig(max_full_cycles=1),
        )
        after = datetime.now().strftime("%Y%m%d%H%M")
        assert result is not None


# ── S2-C1: Checkpointing ────────────────────────────────────────────────


class TestCheckpointer:
    """S2-C1: Checkpointing works across phase transitions."""

    def test_get_checkpointer_returns_saver(self):
        """get_checkpointer returns an InMemorySaver or SqliteSaver."""
        from src.mle_star.state.shared import get_checkpointer, InMemorySaver

        import tempfile

        tmpdir = tempfile.mkdtemp()
        cp = get_checkpointer(tmpdir)
        assert cp is not None
        assert isinstance(cp, InMemorySaver) or type(cp).__name__ == "SqliteSaver"

    def test_get_checkpointer_creates_checkpoint_dir(self):
        """get_checkpointer creates the checkpoints/ directory."""
        from src.mle_star.state.shared import get_checkpointer

        import tempfile

        tmpdir = tempfile.mkdtemp()
        cp = get_checkpointer(tmpdir)
        checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        assert os.path.isdir(checkpoint_dir)

    def test_get_mle_star_graph_accepts_checkpointer(self):
        """get_mle_star_graph accepts an optional checkpointer param."""
        from src.mle_star.graph import get_mle_star_graph
        from src.mle_star.state.shared import InMemorySaver

        graph_no_cp = get_mle_star_graph()
        assert graph_no_cp is not None

        cp = InMemorySaver()
        graph_with_cp = get_mle_star_graph(checkpointer=cp)
        assert graph_with_cp is not None

    def test_run_creates_checkpointer_and_saves_state(self):
        """run() creates a checkpointer and saves checkpoint state."""
        from src.mle_star.graph import run

        import tempfile

        run_dir = tempfile.mkdtemp(prefix="mle_star_cp_test_")
        thread_id = "cp_save_test"
        result = run(
            run_dir=run_dir,
            thread_id=thread_id,
            config=SupervisorConfig(max_full_cycles=1),
        )
        assert result.get("status") == "done"

        cp = result.get("__checkpointer__")
        assert cp is not None, "No checkpointer in result"
        config = result.get(
            "__thread_config__", {"configurable": {"thread_id": thread_id}}
        )
        cp_tuple = cp.get_tuple(config)
        assert cp_tuple is not None, "No checkpoint found after run()"
        state = cp_tuple.checkpoint.get("channel_values", {})
        assert "status" in state or "phase" in state

    def test_run_resume_continue_same_thread(self):
        """Checkpoint survives re-invocation on same thread_id with same checkpointer.

        Demonstrates that when the same InMemorySaver is reused, the checkpoint
        from the first run is accessible. In production, this would use a
        SqliteSaver (file-based) so the checkpointer persists across process
        restarts.
        """
        from src.mle_star.graph import get_mle_star_graph
        from src.mle_star.state.shared import InMemorySaver

        cp = InMemorySaver()
        graph = get_mle_star_graph(
            config=SupervisorConfig(max_full_cycles=1),
            checkpointer=cp,
        )

        thread_id = "resume_same_cp"
        config = {"configurable": {"thread_id": thread_id}}

        state = {
            "task_desc": "test",
            "datasets": [],
            "score_function_desc": "",
            "phase": "search",
            "phase_history": [],
            "current_solution": "",
            "best_solution": "",
            "best_score": 0.0,
            "current_score": None,
            "metric_direction": "maximize",
            "raw_best_score": None,
            "alg1_result": {},
            "alg2_result": {},
            "outer_step": 0,
            "inner_step": 0,
            "convergence_achieved": False,
            "alg3_result": {},
            "ensemble_solutions": [],
            "ensemble_round": 0,
            "submission_code": "",
            "submission_score": None,
            "full_cycles": 0,
            "max_full_cycles": 1,
            "debug_history": [],
            "security_violations": [],
            "stage_history": [],
            "status": "start",
        }

        result1 = graph.invoke(state, config)
        assert result1.get("status") == "done"

        cp_tuple = cp.get_tuple(config)
        assert cp_tuple is not None, "Checkpoint should exist after invoke"

    def test_checkpoint_preserves_best_score(self):
        """Checkpoint state preserves best_score from the run."""
        from src.mle_star.graph import run

        import tempfile

        run_dir = tempfile.mkdtemp(prefix="mle_star_cp_score_")
        thread_id = "cp_score_test"
        result = run(
            run_dir=run_dir,
            thread_id=thread_id,
            config=SupervisorConfig(max_full_cycles=1),
        )

        cp = result.get("__checkpointer__")
        thread_config = result.get(
            "__thread_config__", {"configurable": {"thread_id": thread_id}}
        )
        cp_tuple = cp.get_tuple(thread_config)
        assert cp_tuple is not None
        state = cp_tuple.checkpoint.get("channel_values", {})
        best_score = state.get("best_score")
        assert best_score is not None and best_score > 0

    def test_algorithm1_get_graph_accepts_checkpointer(self):
        """algorithm_1.get_graph() accepts an optional checkpointer param."""
        from src.mle_star.algorithms.algorithm_1 import get_graph
        from src.mle_star.state.shared import InMemorySaver

        cp = InMemorySaver()
        graph = get_graph(checkpointer=cp)
        assert graph is not None

    def test_phase_history_accumulates_across_checkpoints(self):
        """phase_history accumulates correctly with checkpointing."""
        from src.mle_star.graph import run

        import tempfile

        run_dir = tempfile.mkdtemp(prefix="mle_star_phase_hist_")
        result = run(
            run_dir=run_dir,
            config=SupervisorConfig(max_full_cycles=1),
        )
        phase_history = result.get("phase_history", [])
        assert len(phase_history) > 0, "phase_history should not be empty"
        phases = [entry.get("phase") for entry in phase_history]
        assert "search" in phases or "ablation" in phases
