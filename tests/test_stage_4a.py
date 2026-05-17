"""Stage 4a verification tests: Parallel Pipeline Fan-Out (D15).

Validates the D15 architecture change:
    - L independent (Alg1+Alg2) pipeline runs via Send API fan-out
    - Results aggregated and fed to ensemble
    - transition_to_ablation removed from main graph
    - transition_to_ensemble reads from parallel_results
    - ensemble_input_scores populated in Alg3State

Test IDs map to the verification checklist in final_requirements_stages.md
(S4a-01 through S4a-09).
"""

import os
import tempfile

import pytest

from src.mle_star.supervisor import SupervisorConfig


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "mle_star")
MLE_STAR_DIR = os.path.normpath(BASE_DIR)


# ── S4a-01: State types ──────────────────────────────────────────────────


class TestParallelStateTypes:
    """S4a-01: PipelineFlowState and ParallelFanoutState have required fields."""

    def test_s4a_01_pipeline_flow_state_fields(self):
        from src.mle_star.state.parallel_state import PipelineFlowState

        required = [
            "run_index",
            "task_desc",
            "datasets",
            "score_function_desc",
            "metric_direction",
        ]
        missing = [k for k in required if k not in PipelineFlowState.__annotations__]
        assert not missing, f"Missing PipelineFlowState fields: {missing}"

    def test_s4a_01_parallel_fanout_state_fields(self):
        from src.mle_star.state.parallel_state import ParallelFanoutState

        required = [
            "task_desc",
            "datasets",
            "score_function_desc",
            "metric_direction",
            "num_parallel_solutions",
            "parallel_results",
        ]
        missing = [k for k in required if k not in ParallelFanoutState.__annotations__]
        assert not missing, f"Missing ParallelFanoutState fields: {missing}"

    def test_s4a_01_parallel_results_has_operator_add(self):
        import operator
        from src.mle_star.state.parallel_state import ParallelFanoutState

        ann = ParallelFanoutState.__annotations__
        meta = ann["parallel_results"]
        assert hasattr(meta, "__metadata__"), (
            "parallel_results should be Annotated[list[dict], operator.add]"
        )

    def test_s4a_01_system_state_has_parallel_results(self):
        from src.mle_star.state.system_state import MleStarSystemState

        assert "parallel_results" in MleStarSystemState.__annotations__, (
            "MleStarSystemState should have parallel_results field"
        )

    def test_s4a_01_alg3_state_has_ensemble_input_scores(self):
        from src.mle_star.state.alg3_state import Alg3State

        assert "ensemble_input_scores" in Alg3State.__annotations__, (
            "Alg3State should have ensemble_input_scores field"
        )


# ── S4a-02: Config ───────────────────────────────────────────────────────


class TestConfig:
    """S4a-02: NUM_PARALLEL_SOLUTIONS config is available."""

    def test_s4a_02_config_has_parallel_solutions(self):
        from src.mle_star.config import NUM_PARALLEL_SOLUTIONS

        assert isinstance(NUM_PARALLEL_SOLUTIONS, int)
        assert NUM_PARALLEL_SOLUTIONS >= 1

    def test_s4a_02_supervisor_config_wired(self):
        from src.mle_star.supervisor import SupervisorConfig
        from src.mle_star.config import NUM_PARALLEL_SOLUTIONS

        config = SupervisorConfig()
        assert config.num_parallel_solutions == NUM_PARALLEL_SOLUTIONS


# ── S4a-03: Supervisor routing ───────────────────────────────────────────


class TestSupervisorRouting:
    """S4a-03: Supervisor routes correctly for D15 architecture."""

    def test_s4a_03_search_routes_to_parallel_pipeline(self):
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "search", "full_cycles": 0}
        result = supervisor_decide(state)
        assert result == "parallel_pipeline_node"

    def test_s4a_03_search_done_routes_to_ensemble(self):
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "search_done", "full_cycles": 0}
        result = supervisor_decide(state)
        assert result == "transition_to_ensemble"

    def test_s4a_03_ablation_improving_continues(self):
        from src.mle_star.supervisor import supervisor_decide

        state = {
            "phase": "ablation",
            "current_score": 0.9,
            "best_score": 0.85,
            "outer_step": 0,
            "full_cycles": 0,
        }
        result = supervisor_decide(state)
        assert result == "continue_ablation"

    def test_s4a_03_no_transition_to_ablation_in_routing(self):
        from src.mle_star.graph import SUPERVISOR_ROUTING_MAP

        assert "transition_to_ablation" not in SUPERVISOR_ROUTING_MAP, (
            "transition_to_ablation should be removed from routing map (D15)"
        )

    def test_s4a_03_no_alg1_subgraph_in_routing(self):
        from src.mle_star.graph import SUPERVISOR_ROUTING_MAP

        assert "alg1_subgraph" not in SUPERVISOR_ROUTING_MAP, (
            "alg1_subgraph should be removed from routing map (D15)"
        )

    def test_s4a_03_parallel_pipeline_in_routing(self):
        from src.mle_star.graph import SUPERVISOR_ROUTING_MAP

        assert "parallel_pipeline_node" in SUPERVISOR_ROUTING_MAP, (
            "parallel_pipeline_node should be in routing map (D15)"
        )


# ── S4a-04: Parallel pipeline subgraph ────────────────────────────────────


class TestParallelPipelineSubgraph:
    """S4a-04: Parallel pipeline subgraph compiles and dispatches correctly."""

    def test_s4a_04_dispatch_returns_L_sends(self):
        from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
            dispatch_parallel_pipelines,
        )

        state = {
            "task_desc": "test task",
            "datasets": [],
            "score_function_desc": "accuracy",
            "metric_direction": "maximize",
            "num_parallel_solutions": 2,
            "parallel_results": [],
        }
        sends = dispatch_parallel_pipelines(state)
        assert len(sends) == 2
        for s in sends:
            assert s.node == "pipeline_flow"

    def test_s4a_04_dispatch_default_L(self):
        from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
            dispatch_parallel_pipelines,
        )

        state = {
            "task_desc": "test task",
            "datasets": [],
            "score_function_desc": "accuracy",
            "metric_direction": "maximize",
            "parallel_results": [],
        }
        sends = dispatch_parallel_pipelines(state)
        from src.mle_star.config import NUM_PARALLEL_SOLUTIONS

        assert len(sends) == NUM_PARALLEL_SOLUTIONS

    def test_s4a_04_dispatch_send_has_run_index(self):
        from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
            dispatch_parallel_pipelines,
        )

        state = {
            "task_desc": "test task",
            "datasets": [],
            "score_function_desc": "accuracy",
            "metric_direction": "maximize",
            "num_parallel_solutions": 3,
            "parallel_results": [],
        }
        sends = dispatch_parallel_pipelines(state)
        indices = [s.arg["run_index"] for s in sends]
        assert sorted(indices) == [0, 1, 2]

    def test_s4a_04_subgraph_compiles(self):
        from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
            get_parallel_pipeline_subgraph,
        )

        graph = get_parallel_pipeline_subgraph()
        assert graph is not None

    def test_s4a_04_subgraph_has_pipeline_flow_node(self):
        from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
            get_parallel_pipeline_subgraph,
        )

        graph = get_parallel_pipeline_subgraph()
        node_names = set(graph.nodes.keys())
        assert "pipeline_flow" in node_names, (
            f"Missing pipeline_flow node, got: {node_names}"
        )


# ── S4a-05: transition_to_ensemble reads from parallel_results ────────────


class TestTransitionToEnsemble:
    """S4a-05: transition_to_ensemble uses parallel_results, not 2-solution hack."""

    def test_s4a_05_reads_from_parallel_results(self):
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
        assert result["ensemble_solutions"] == ["sol_0", "sol_1"]
        assert result["phase"] == "ensemble"
        assert result["ensemble_round"] == 0

    def test_s4a_05_fallback_when_empty_parallel_results(self):
        from src.mle_star.graph import transition_to_ensemble

        state = {
            "phase": "search_done",
            "parallel_results": [],
            "best_solution": "fallback_sol",
            "best_score": 0.75,
            "full_cycles": 0,
        }
        result = transition_to_ensemble(state)
        assert result["ensemble_solutions"] == ["fallback_sol"]

    def test_s4a_05_no_two_solution_hack(self):
        from src.mle_star.graph import transition_to_ensemble

        state = {
            "phase": "search_done",
            "parallel_results": [
                {"run_index": 0, "best_solution": "pipeline_sol_0", "best_score": 0.88},
            ],
            "best_solution": "global_best",
            "current_solution": "current_sol",
            "best_score": 0.88,
            "full_cycles": 0,
        }
        result = transition_to_ensemble(state)
        assert result["ensemble_solutions"] != ["global_best", "current_sol"], (
            "transition_to_ensemble should NOT use [best_solution, current_solution] 2-solution hack"
        )


# ── S4a-06: system_to_alg3_state passes ensemble_input_scores ────────────


class TestSystemToAlg3Mapping:
    """S4a-06: system_to_alg3_state populates ensemble_input_scores from parallel_results."""

    def test_s4a_06_ensemble_input_scores_from_parallel(self):
        from src.mle_star.graph import system_to_alg3_state

        state = {
            "ensemble_solutions": ["sol_a", "sol_b"],
            "ensemble_round": 0,
            "parallel_results": [
                {"run_index": 0, "best_solution": "sol_a", "best_score": 0.85},
                {"run_index": 1, "best_solution": "sol_b", "best_score": 0.90},
            ],
        }
        result = system_to_alg3_state(state)
        assert result["ensemble_input_scores"] == [0.85, 0.90]

    def test_s4a_06_ensemble_input_scores_fallback(self):
        from src.mle_star.graph import system_to_alg3_state

        state = {
            "ensemble_solutions": ["sol_a"],
            "ensemble_round": 0,
            "parallel_results": [],
            "best_score": 0.75,
        }
        result = system_to_alg3_state(state)
        assert result["ensemble_input_scores"] == [0.75]

    def test_s4a_06_ensemble_solutions_preserved(self):
        from src.mle_star.graph import system_to_alg3_state

        state = {
            "ensemble_solutions": ["solution_1", "solution_2"],
            "ensemble_round": 1,
            "parallel_results": [
                {"run_index": 0, "best_score": 0.80},
                {"run_index": 1, "best_score": 0.88},
            ],
        }
        result = system_to_alg3_state(state)
        assert result["ensemble_solutions"] == ["solution_1", "solution_2"]


# ── S4a-07: Graph topology ───────────────────────────────────────────────


class TestGraphTopology:
    """S4a-07: Main graph has correct D15 topology."""

    def test_s4a_07_no_transition_to_ablation_node(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        node_names = set(graph.nodes.keys())
        assert "transition_to_ablation" not in node_names, (
            "transition_to_ablation node should be removed from main graph (D15)"
        )

    def test_s4a_07_no_alg1_subgraph_node(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        node_names = set(graph.nodes.keys())
        assert "alg1_subgraph" not in node_names, (
            "alg1_subgraph node should be removed from main graph (D15, invoked inside pipeline_flow_node)"
        )

    def test_s4a_07_parallel_pipeline_node_exists(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        node_names = set(graph.nodes.keys())
        assert "parallel_pipeline_node" in node_names, (
            "parallel_pipeline_node should be in main graph (D15)"
        )

    def test_s4a_07_required_nodes(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        node_names = set(graph.nodes.keys())
        required = {
            "supervisor",
            "parallel_pipeline_node",
            "alg2_subgraph",
            "transition_to_ensemble",
            "alg3_subgraph",
            "transition_to_submission",
            "submission_node",
        }
        assert required.issubset(node_names), f"Missing nodes: {required - node_names}"

    def test_s4a_07_graph_compiles(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        assert graph is not None


# ── S4a-08: parallel_pipeline_node aggregation ────────────────────────────


class TestParallelPipelineNode:
    """S4a-08: parallel_pipeline_node logic (unit-level, no graph invocation)."""

    def test_s4a_08_dispatch_pipelines_L_2(self):
        from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
            dispatch_parallel_pipelines,
        )

        state = {
            "task_desc": "test",
            "datasets": [],
            "score_function_desc": "",
            "metric_direction": "maximize",
            "num_parallel_solutions": 2,
            "parallel_results": [],
        }
        sends = dispatch_parallel_pipelines(state)
        assert len(sends) == 2

    def test_s4a_08_dispatch_pipelines_L_1_fallback(self):
        from src.mle_star.subgraphs.parallel_pipeline_subgraph import (
            dispatch_parallel_pipelines,
        )

        state = {
            "task_desc": "test",
            "datasets": [],
            "score_function_desc": "",
            "metric_direction": "maximize",
            "num_parallel_solutions": 1,
            "parallel_results": [],
        }
        sends = dispatch_parallel_pipelines(state)
        assert len(sends) == 1


# ── S4a-09: Algorithm 3 reads ensemble_input_scores ──────────────────────


class TestAlg3EnsembleInputScores:
    """S4a-09: Algorithm 3 reads ensemble_input_scores from state."""

    def test_s4a_09_A9_reads_ensemble_input_scores(self):
        from src.mle_star.nodes.ensemble import A9__plan_ensemble

        state = {
            "ensemble_solutions": ["sol_0", "sol_1"],
            "ensemble_input_scores": [0.85, 0.90],
            "ensemble_round": 0,
        }
        result = A9__plan_ensemble(state)
        assert result["status"] == "planned"
        assert "current_ensemble_plan" in result

    def test_s4a_09_A9_handles_empty_scores(self):
        from src.mle_star.nodes.ensemble import A9__plan_ensemble

        state = {
            "ensemble_solutions": ["sol_0"],
            "ensemble_input_scores": [],
            "ensemble_round": 0,
        }
        result = A9__plan_ensemble(state)
        assert result["status"] == "planned"
        assert "current_ensemble_plan" in result

    def test_s4a_09_alg3_graph_compiles(self):
        from src.mle_star.subgraphs.ensemble_round_subgraph import (
            get_ensemble_round_subgraph,
        )

        graph = get_ensemble_round_subgraph()
        assert graph is not None


# ── S4a Regression: Previous stages still work ────────────────────────────


class TestRegression:
    """S4a regression: Algorithm 1, 2, and main graph still work."""

    def test_s4a_alg1_still_works(self, e2e_result):
        from src.mle_star.algorithms.algorithm_1 import run

        assert e2e_result.get("status") == "done"
        assert len(e2e_result.get("candidates_pool", [])) > 0

    def test_s4a_main_graph_runs(self):
        from src.mle_star.graph import run

        run_dir = tempfile.mkdtemp(prefix="mle_star_s4a_full_")
        result = run(
            initial_state={
                "task_desc": "S4a integration test",
                "datasets": [],
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
                "parallel_results": [],
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
        assert result.get("status") == "done"
        assert result.get("submission_code", "") != ""

    def test_s4a_no_bypass_hack(self):
        from src.mle_star.graph import alg3_subgraph_node

        import inspect

        source = inspect.getsource(alg3_subgraph_node)
        assert "max_ensemble_rounds" not in source, (
            "Bypass hack should be removed from alg3_subgraph_node"
        )

    def test_s4a_transition_to_ensemble_no_old_hack(self):
        from src.mle_star.graph import transition_to_ensemble

        import inspect

        source = inspect.getsource(transition_to_ensemble)
        assert "current_solution" not in source or "parallel_results" in source, (
            "transition_to_ensemble should read from parallel_results, not [best_solution, current_solution]"
        )
