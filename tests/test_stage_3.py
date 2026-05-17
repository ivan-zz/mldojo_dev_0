"""Stage 3 verification tests: Ablation Topology (revised architecture).

Validates Algorithm 2's three-layer nested architecture:
    Layer 1: algorithms/algorithm_2.py — Python for loop with SubgraphSpan
    Layer 2: subgraphs/ablation_subgraph.py — single-pass, Send API fan-out
    Layer 3: subgraphs/refinement_subgraph.py — inner loop with conditional edges

Also validates:
    - Ablation variant subgraph (eval + debug retry loop)
    - Separate script execution per variant (not one monolithic script)
    - Narrow state types (AblationCycleState, AblationVariantState, RefinementInnerState)
    - State mapping functions between MleStarSystemState and Alg2State

Test IDs map to the verification checklist in stage_verification.md
(S3-01 through S3-16, updated for revised architecture).
"""

import os
import tempfile

import pytest

from src.mle_star.supervisor import SupervisorConfig


BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "mle_star")
MLE_STAR_DIR = os.path.normpath(BASE_DIR)


# ── S3-01: State type fields ──────────────────────────────────────────────


class TestStateTypes:
    """S3-01: All state types have required fields."""

    def test_s3_01_alg2_state_fields(self):
        from src.mle_star.state.alg2_state import Alg2State

        required = [
            "current_solution",
            "best_score",
            "metric_direction",
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
            "improved_solution",
            "improved_score",
            "convergence_achieved",
            "stage_history",
            "status",
        ]
        missing = [k for k in required if k not in Alg2State.__annotations__]
        assert not missing, f"Missing Alg2State fields: {missing}"

    def test_s3_01_ablation_cycle_state_fields(self):
        from src.mle_star.state.alg2_state import AblationCycleState

        required = [
            "current_solution",
            "best_score",
            "metric_direction",
            "ablation_scripts",
            "functional_blocks",
            "ablation_results_list",
            "previous_summaries",
            "previous_blocks",
            "target_block",
            "initial_plan",
            "status",
        ]
        missing = [k for k in required if k not in AblationCycleState.__annotations__]
        assert not missing, f"Missing AblationCycleState fields: {missing}"

    def test_s3_01_ablation_variant_state_fields(self):
        from src.mle_star.state.alg2_state import AblationVariantState

        required = [
            "variant_name",
            "variant_code",
            "block_name",
            "execution_output",
            "execution_error",
            "execution_score",
            "attempts",
            "status",
        ]
        missing = [k for k in required if k not in AblationVariantState.__annotations__]
        assert not missing, f"Missing AblationVariantState fields: {missing}"

    def test_s3_01_refinement_inner_state_fields(self):
        from src.mle_star.state.alg2_state import RefinementInnerState

        required = [
            "target_block",
            "current_solution",
            "best_score",
            "initial_plan",
            "current_plan",
            "refined_code",
            "candidate_solution",
            "execution_output",
            "execution_error",
            "execution_score",
            "inner_step",
            "debug_retries",
            "improved_solution",
            "improved_score",
            "status",
        ]
        missing = [k for k in required if k not in RefinementInnerState.__annotations__]
        assert not missing, f"Missing RefinementInnerState fields: {missing}"

    def test_s3_01_ablation_cycle_state_operator_add(self):
        from src.mle_star.state.alg2_state import AblationCycleState
        import operator

        ann = AblationCycleState.__annotations__
        assert "ablation_results_list" in ann
        meta = ann["ablation_results_list"]
        assert hasattr(meta, "__metadata__"), (
            "ablation_results_list should be Annotated"
        )

    def test_s3_01_ablation_fanout_state_fields(self):
        from src.mle_star.state.alg2_state import AblationFanoutState

        required = ["current_solution", "ablation_scripts", "ablation_results"]
        missing = [k for k in required if k not in AblationFanoutState.__annotations__]
        assert not missing, f"Missing AblationFanoutState fields: {missing}"


# ── S3-02: Mock ablation nodes exist ─────────────────────────────────────


class TestAblationNodes:
    """S3-02: nodes/ablation.py contains mock nodes."""

    def test_s3_02_A4_generate_ablation(self):
        from src.mle_star.nodes.ablation import A4__generate_ablation

        assert callable(A4__generate_ablation)

    def test_s3_02_A5_summarize_ablation(self):
        from src.mle_star.nodes.ablation import A5__summarize_ablation

        assert callable(A5__summarize_ablation)

    def test_s3_02_A6_extract_block(self):
        from src.mle_star.nodes.ablation import A6__extract_block

        assert callable(A6__extract_block)


# ── S3-02b: Mock refinement nodes exist ───────────────────────────────────


class TestRefinementNodes:
    """S3-02b: nodes/refinement.py contains mock nodes."""

    def test_s3_02b_A7_implement(self):
        from src.mle_star.nodes.refinement import A7__implement

        assert callable(A7__implement)

    def test_s3_02b_A_verify(self):
        from src.mle_star.nodes.refinement import A_verify

        assert callable(A_verify)

    def test_s3_02b_A_sast(self):
        from src.mle_star.nodes.refinement import A_sast

        assert callable(A_sast)

    def test_s3_02b_eval_refinement(self):
        from src.mle_star.nodes.refinement import eval_refinement

        assert callable(eval_refinement)

    def test_s3_02b_A8_plan(self):
        from src.mle_star.nodes.refinement import A8__plan

        assert callable(A8__plan)

    def test_s3_02b_A11_debug_refine(self):
        from src.mle_star.nodes.refinement import A11__debug_refine

        assert callable(A11__debug_refine)

    def test_s3_02b_route_functions_callable(self):
        from src.mle_star.nodes.refinement import (
            route_after_verify,
            route_after_sast,
            route_after_eval_step,
        )

        assert callable(route_after_verify)
        assert callable(route_after_sast)
        assert callable(route_after_eval_step)

    def test_s3_02b_route_after_eval_step(self):
        from src.mle_star.nodes.refinement import route_after_eval_step
        from langgraph.graph import END

        ok_state = {"status": "ok", "debug_retries": 0}
        assert route_after_eval_step(ok_state) == END

        error_retryable = {"status": "error", "debug_retries": 0}
        assert route_after_eval_step(error_retryable) == "A11__debug_refine"

        error_maxed = {"status": "error", "debug_retries": 3}
        assert route_after_eval_step(error_maxed) == END


# ── S3-03: Subgraphs compile ─────────────────────────────────────────────


class TestSubgraphCompilation:
    """S3-03: All subgraphs compile correctly."""

    def test_s3_03_ablation_subgraph_compiles(self):
        from src.mle_star.subgraphs.ablation_subgraph import get_ablation_subgraph

        graph = get_ablation_subgraph()
        assert graph is not None

    def test_s3_03_ablation_variant_subgraph_compiles(self):
        from src.mle_star.subgraphs.ablation_variant_subgraph import (
            ablation_variant_subgraph,
        )

        assert ablation_variant_subgraph is not None

    def test_s3_03_refinement_subgraph_compiles(self):
        from src.mle_star.subgraphs.refinement_subgraph import get_refinement_subgraph

        graph = get_refinement_subgraph()
        assert graph is not None

    def test_s3_03_ablation_subgraph_nodes(self):
        from src.mle_star.subgraphs.ablation_subgraph import get_ablation_subgraph

        graph = get_ablation_subgraph()
        node_names = set(graph.nodes.keys())
        expected = {
            "A4__generate_ablation",
            "ablation_variant_flow",
            "A5__summarize_ablation",
            "A6__extract_block",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_s3_03_ablation_variant_subgraph_nodes(self):
        from src.mle_star.subgraphs.ablation_variant_subgraph import (
            ablation_variant_subgraph,
        )

        node_names = set(ablation_variant_subgraph.nodes.keys())
        expected = {"eval_ablation_variant", "A11__debug_ablation_variant"}
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_s3_03_dispatch_returns_correct_send_count(self):
        from src.mle_star.subgraphs.ablation_subgraph import _dispatch_from_cycle_state
        from langgraph.types import Send

        scripts = [
            {"name": "baseline", "code": "print(1)", "block_name": "baseline"},
            {"name": "ablation_0", "code": "print(2)", "block_name": "block_0"},
            {"name": "ablation_1", "code": "print(3)", "block_name": "block_1"},
            {"name": "ablation_2", "code": "print(4)", "block_name": "block_2"},
        ]
        state = {"ablation_scripts": scripts}
        sends = _dispatch_from_cycle_state(state)
        assert len(sends) == 4, f"Expected 4 Send objects, got {len(sends)}"
        for s in sends:
            assert isinstance(s, Send), f"Expected Send, got {type(s)}"
            assert s.node == "ablation_variant_flow"

    def test_s3_03_refinement_subgraph_nodes(self):
        from src.mle_star.subgraphs.refinement_subgraph import (
            get_refinement_step_subgraph,
        )

        graph = get_refinement_step_subgraph()
        node_names = set(graph.nodes.keys())
        expected = {
            "A7__implement",
            "A_verify",
            "A_sast",
            "eval_refinement",
            "A11__debug_refine",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


# ── S3-04: A4__generate_ablation returns ablation_scripts ─────────────────


class TestA4GenerateAblation:
    """S3-04: A4__generate_ablation returns state with ablation_scripts list."""

    def test_s3_04_returns_ablation_scripts(self):
        from src.mle_star.nodes.ablation import A4__generate_ablation

        state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
        }
        result = A4__generate_ablation(state)
        assert "ablation_scripts" in result
        assert isinstance(result["ablation_scripts"], list)
        assert len(result["ablation_scripts"]) >= 1

    def test_s3_04_first_script_is_baseline(self):
        from src.mle_star.nodes.ablation import A4__generate_ablation

        state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
        }
        result = A4__generate_ablation(state)
        scripts = result["ablation_scripts"]
        assert scripts[0]["name"] == "baseline"
        assert scripts[0]["block_name"] == "baseline"

    def test_s3_04_variant_scripts_have_block_names(self):
        from src.mle_star.nodes.ablation import A4__generate_ablation

        state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
        }
        result = A4__generate_ablation(state)
        scripts = result["ablation_scripts"]
        variant_scripts = [s for s in scripts if s["block_name"] != "baseline"]
        assert len(variant_scripts) >= 1
        for s in variant_scripts:
            assert "block_name" in s
            assert s["block_name"] != "baseline"

    def test_s3_04_returns_status(self):
        from src.mle_star.nodes.ablation import A4__generate_ablation

        state = {"current_solution": "", "best_score": 0.85}
        result = A4__generate_ablation(state)
        assert "status" in result
        assert result["status"] == "ablation_generated"

    def test_s3_04_returns_functional_blocks(self):
        from src.mle_star.nodes.ablation import A4__generate_ablation

        state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
        }
        result = A4__generate_ablation(state)
        assert "functional_blocks" in result
        assert isinstance(result["functional_blocks"], list)


# ── S3-05: A5__summarize_ablation returns ablation_summaries ────────────


class TestA5SummarizeAblation:
    """S3-05: A5__summarize_ablation returns accumulated summaries."""

    def test_s3_05_returns_ablation_summaries(self):
        from src.mle_star.nodes.ablation import A5__summarize_ablation

        state = {
            "ablation_results_list": [
                {"block_name": "baseline", "execution_score": 0.85, "impact": 0},
                {"block_name": "preprocess", "masked_score": 0.75, "impact": 0.10},
                {"block_name": "train", "masked_score": 0.80, "impact": 0.05},
            ],
        }
        result = A5__summarize_ablation(state)
        assert "ablation_summaries" in result
        assert isinstance(result["ablation_summaries"], list)
        assert len(result["ablation_summaries"]) == 1
        assert "preprocess" in result["ablation_summaries"][0]

    def test_s3_05_summary_identifies_highest_impact(self):
        from src.mle_star.nodes.ablation import A5__summarize_ablation

        state = {
            "ablation_results_list": [
                {"block_name": "baseline", "execution_score": 0.85, "impact": 0},
                {"block_name": "feature_eng", "masked_score": 0.70, "impact": 0.15},
                {"block_name": "preprocess", "masked_score": 0.80, "impact": 0.05},
                {"block_name": "train", "masked_score": 0.83, "impact": 0.02},
            ],
        }
        result = A5__summarize_ablation(state)
        assert "feature_eng" in result["ablation_summaries"][0]


# ── S3-06: A6__extract_block returns target_block and initial_plan ───────


class TestA6ExtractBlock:
    """S3-06: A6__extract_block returns target_block and initial_plan."""

    def test_s3_06_returns_target_block(self):
        from src.mle_star.nodes.ablation import A6__extract_block

        state = {
            "ablation_results_list": [
                {"block_name": "baseline", "execution_score": 0.85, "impact": 0},
                {"block_name": "feature_eng", "masked_score": 0.70, "impact": 0.15},
                {"block_name": "preprocess", "masked_score": 0.80, "impact": 0.05},
            ],
            "functional_blocks": [
                {
                    "name": "feature_eng",
                    "code": "def feature_eng(df): pass",
                    "start_line": 1,
                    "end_line": 2,
                    "type": "FunctionDef",
                },
                {
                    "name": "preprocess",
                    "code": "def preprocess(df): pass",
                    "start_line": 4,
                    "end_line": 5,
                    "type": "FunctionDef",
                },
            ],
        }
        result = A6__extract_block(state)
        assert "target_block" in result
        assert isinstance(result["target_block"], str)
        assert "feature_eng" in result["target_block"]

    def test_s3_06_returns_initial_plan(self):
        from src.mle_star.nodes.ablation import A6__extract_block

        state = {
            "ablation_results_list": [
                {"block_name": "baseline", "execution_score": 0.85, "impact": 0},
                {"block_name": "train", "masked_score": 0.75, "impact": 0.10},
            ],
            "functional_blocks": [
                {
                    "name": "train",
                    "code": "def train(X, y): pass",
                    "start_line": 1,
                    "end_line": 2,
                    "type": "FunctionDef",
                },
            ],
        }
        result = A6__extract_block(state)
        assert "initial_plan" in result
        assert isinstance(result["initial_plan"], str)
        assert len(result["initial_plan"]) > 0

    def test_s3_06_respects_previous_blocks(self):
        from src.mle_star.nodes.ablation import A6__extract_block

        state = {
            "ablation_results_list": [
                {"block_name": "baseline", "execution_score": 0.85, "impact": 0},
                {"block_name": "feature_eng", "masked_score": 0.70, "impact": 0.15},
                {"block_name": "preprocess", "masked_score": 0.80, "impact": 0.05},
            ],
            "functional_blocks": [
                {
                    "name": "feature_eng",
                    "code": "def feature_eng(df): pass",
                    "start_line": 1,
                    "end_line": 2,
                    "type": "FunctionDef",
                },
                {
                    "name": "preprocess",
                    "code": "def preprocess(df): pass",
                    "start_line": 4,
                    "end_line": 5,
                    "type": "FunctionDef",
                },
            ],
            "previous_blocks": ["feature_eng"],
        }
        result = A6__extract_block(state)
        assert "preprocess" in result["target_block"]


# ── S3-07: Ablation variant subgraph works ────────────────────────────────


class TestAblationVariantSubgraph:
    """S3-07: Ablation variant subgraph executes and handles debug retries."""

    def test_s3_07_variant_subgraph_runs(self):
        from src.mle_star.subgraphs.ablation_variant_subgraph import (
            ablation_variant_subgraph,
        )

        state = {
            "variant_name": "baseline",
            "variant_code": "# test code",
            "block_name": "baseline",
            "execution_output": "",
            "execution_error": None,
            "execution_score": None,
            "attempts": 0,
            "status": "pending",
        }
        result = ablation_variant_subgraph.invoke(state)
        assert result is not None
        assert result.get("status") == "ok"
        assert result.get("execution_score") is not None


# ── S3-08: Refinement nodes work ─────────────────────────────────────────


class TestRefinementNodes:
    """S3-08: Refinement nodes return expected output."""

    def test_s3_08_A7_implement(self):
        from src.mle_star.nodes.refinement import A7__implement

        state = {
            "current_plan": "Improve preprocessing",
            "initial_plan": "Improve preprocessing",
            "target_block": "def preprocess(df):\n    return df\n",
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X,y): pass\n",
        }
        result = A7__implement(state)
        assert "refined_code" in result
        assert "candidate_solution" in result
        assert result["status"] == "implemented"

    def test_s3_08_A_verify_passes(self):
        from src.mle_star.nodes.refinement import A_verify

        state = {"refined_code": "def test(): pass"}
        result = A_verify(state)
        assert result["status"] == "ok"

    def test_s3_08_A_sast_passes(self):
        from src.mle_star.nodes.refinement import A_sast

        state = {"refined_code": "def test(): pass"}
        result = A_sast(state)
        assert result["status"] == "pass"

    def test_s3_08_eval_refinement(self):
        from src.mle_star.nodes.refinement import eval_refinement

        state = {
            "candidate_solution": "def test(): pass",
            "best_score": 0.85,
            "debug_retries": 0,
            "inner_step": 0,
        }
        result = eval_refinement(state)
        assert "status" in result
        assert "execution_score" in result

    def test_s3_08_A8_plan(self):
        from src.mle_star.nodes.refinement import A8__plan

        state = {
            "target_block": "def preprocess(df):\n    return df\n",
            "inner_step": 0,
            "execution_score": 0.87,
        }
        result = A8__plan(state)
        assert "current_plan" in result
        assert "inner_step" in result
        assert result["inner_step"] == 1


# ── S3-09: Refinement step subgraph runs single pass ────────────────────────


class TestRefinementStepSubgraph:
    """S3-09: Refinement step subgraph runs a single pass (A7→verify→sast→eval)."""

    def test_s3_09_refinement_step_subgraph_runs(self):
        from src.mle_star.subgraphs.refinement_subgraph import (
            get_refinement_step_subgraph,
        )

        graph = get_refinement_step_subgraph()
        state = {
            "target_block": "def preprocess(df):\n    return df\n",
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X,y): pass\n",
            "best_score": 0.85,
            "initial_plan": "Improve preprocessing",
            "current_plan": "Improve preprocessing",
            "refined_code": "",
            "candidate_solution": "",
            "execution_output": "",
            "execution_error": None,
            "execution_score": None,
            "inner_step": 0,
            "debug_retries": 0,
            "improved_solution": "",
            "improved_score": 0.0,
            "status": "start",
        }
        result = graph.invoke(state)
        assert result is not None
        assert result.get("status") in ("ok", "error")

    def test_s3_09_refinement_subgraph_alias(self):
        from src.mle_star.subgraphs.refinement_subgraph import (
            get_refinement_step_subgraph,
            get_refinement_subgraph,
        )

        step_graph = get_refinement_step_subgraph()
        alias_graph = get_refinement_subgraph()
        assert step_graph is alias_graph


# ── S3-10: Ablation subgraph runs single-pass ──────────────────────────


class TestAblationSubgraph:
    """S3-10: Ablation subgraph runs single-pass with Send API fan-out."""

    def test_s3_10_ablation_subgraph_runs(self):
        from src.mle_star.subgraphs.ablation_subgraph import get_ablation_subgraph

        graph = get_ablation_subgraph()
        state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
            "ablation_scripts": [],
            "functional_blocks": [],
            "ablation_results_list": [],
            "previous_summaries": [],
            "previous_blocks": [],
            "target_block": "",
            "initial_plan": "",
            "status": "start",
        }
        result = graph.invoke(state)
        assert result is not None
        assert "ablation_results_list" in result
        assert len(result.get("ablation_results_list", [])) >= 1

    def test_s3_10_ablation_subgraph_extracts_block(self):
        from src.mle_star.subgraphs.ablation_subgraph import get_ablation_subgraph

        graph = get_ablation_subgraph()
        state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
            "ablation_scripts": [],
            "functional_blocks": [],
            "ablation_results_list": [],
            "previous_summaries": [],
            "previous_blocks": [],
            "target_block": "",
            "initial_plan": "",
            "status": "start",
        }
        result = graph.invoke(state)
        assert result.get("target_block", "") != ""
        assert result.get("initial_plan", "") != ""


# ── S3-11: Algorithm 2 runs with Python-level outer loop ──────────────


class TestAlg2OuterLoop:
    """S3-11: Algorithm 2 runs with Python-level for loop."""

    def test_s3_11_alg2_runs(self):
        from src.mle_star.algorithms.algorithm_2 import run

        run_dir = tempfile.mkdtemp(prefix="mle_star_alg2_test_")
        initial_state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
            "ablation_scripts": [],
            "ablation_results_list": [],
            "ablation_summaries": [],
            "target_block": "",
            "initial_plan": "",
            "refined_blocks": [],
            "current_plans": [],
            "current_scores": [],
            "refined_code": "",
            "candidate_solution": "",
            "execution_output": "",
            "execution_error": None,
            "execution_score": None,
            "outer_step": 0,
            "inner_step": 0,
            "leakage_status": None,
            "leakage_code_block": None,
            "debug_history": [],
            "security_violations": [],
            "improved_solution": "",
            "improved_score": 0,
            "stage_history": [],
            "status": "start",
        }
        result = run(initial_state=initial_state, run_dir=run_dir)
        assert result is not None
        assert result.get("status") == "done"

    def test_s3_11_ablation_summaries_accumulate(self):
        from src.mle_star.algorithms.algorithm_2 import run

        run_dir = tempfile.mkdtemp(prefix="mle_star_s3_11_")
        initial_state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
            "ablation_scripts": [],
            "ablation_results_list": [],
            "ablation_summaries": [],
            "target_block": "",
            "initial_plan": "",
            "refined_blocks": [],
            "current_plans": [],
            "current_scores": [],
            "refined_code": "",
            "candidate_solution": "",
            "execution_output": "",
            "execution_error": None,
            "execution_score": None,
            "outer_step": 0,
            "inner_step": 0,
            "leakage_status": None,
            "leakage_code_block": None,
            "debug_history": [],
            "security_violations": [],
            "improved_solution": "",
            "improved_score": 0,
            "stage_history": [],
            "status": "start",
        }
        result = run(initial_state=initial_state, run_dir=run_dir)
        summaries = result.get("ablation_summaries", [])
        assert len(summaries) >= 1, (
            "ablation_summaries should accumulate across outer steps"
        )

    def test_s3_11_refined_blocks_accumulate(self):
        from src.mle_star.algorithms.algorithm_2 import run

        run_dir = tempfile.mkdtemp(prefix="mle_star_s3_11b_")
        initial_state = {
            "current_solution": "def preprocess(df):\n    return df\n\ndef train(X, y):\n    return model\n",
            "best_score": 0.85,
            "ablation_scripts": [],
            "ablation_results_list": [],
            "ablation_summaries": [],
            "target_block": "",
            "initial_plan": "",
            "refined_blocks": [],
            "current_plans": [],
            "current_scores": [],
            "refined_code": "",
            "candidate_solution": "",
            "execution_output": "",
            "execution_error": None,
            "execution_score": None,
            "outer_step": 0,
            "inner_step": 0,
            "leakage_status": None,
            "leakage_code_block": None,
            "debug_history": [],
            "security_violations": [],
            "improved_solution": "",
            "improved_score": 0,
            "stage_history": [],
            "status": "start",
        }
        result = run(initial_state=initial_state, run_dir=run_dir)
        blocks = result.get("refined_blocks", [])
        assert len(blocks) >= 1, "refined_blocks should accumulate across outer steps"


# ── S3-12: system_to_alg2_state mapping ─────────────────────────────────


class TestSystemToAlg2Mapping:
    """S3-12: system_to_alg2_state preserves current_solution and best_score."""

    def test_s3_12_mapping_preserves_solution(self):
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

    def test_s3_12_mapping_initializes_clean(self):
        from src.mle_star.graph import system_to_alg2_state

        state = {
            "current_solution": "solution",
            "best_score": 0.9,
            "outer_step": 0,
            "inner_step": 0,
        }
        result = system_to_alg2_state(state)
        assert result["ablation_scripts"] == []
        assert result["ablation_results_list"] == []
        assert result["ablation_summaries"] == []
        assert result["target_block"] == ""
        assert result["initial_plan"] == ""
        assert result["refined_blocks"] == []
        assert result["status"] == "start"

    def test_s3_12_no_ablation_code_field(self):
        from src.mle_star.graph import system_to_alg2_state

        state = {
            "current_solution": "solution",
            "best_score": 0.9,
            "outer_step": 0,
            "inner_step": 0,
        }
        result = system_to_alg2_state(state)
        assert "ablation_code" not in result
        assert "ablation_results" not in result


# ── S3-13: Main graph routes ablation to Algorithm 2 ──────────────────────


class TestGraphAblationRouting:
    """S3-13: Main graph routes phase='ablation' to Algorithm 2 subgraph."""

    def test_s3_13_ablation_routes_correctly(self):
        from src.mle_star.supervisor import supervisor_decide

        state = {
            "phase": "ablation",
            "current_score": 0.8,
            "best_score": 0.8,
            "outer_step": 0,
            "full_cycles": 0,
        }
        result = supervisor_decide(state)
        assert result in ("continue_ablation", "new_ablation_cycle")


# ── S3-14: Alg2 results feed back to MleStarSystemState ──────────────────


class TestAlg2ToSystemState:
    """S3-14: Algorithm 2 results feed back to MleStarSystemState."""

    def test_s3_14_improved_solution_updates_system(self):
        from src.mle_star.graph import alg2_result_to_system

        result = {
            "improved_solution": "def improved(): pass",
            "improved_score": 0.92,
            "outer_step": 3,
            "inner_step": 0,
            "stage_history": [],
            "convergence_achieved": True,
        }
        state = {
            "best_solution": "def old(): pass",
            "best_score": 0.85,
        }
        updates = alg2_result_to_system(result, state)
        assert updates["best_solution"] == "def improved(): pass"
        assert updates["best_score"] == 0.92
        assert updates["current_solution"] == "def improved(): pass"

    def test_s3_14_no_improvement_keeps_old_best(self):
        from src.mle_star.graph import alg2_result_to_system

        result = {
            "improved_solution": "def worse(): pass",
            "improved_score": 0.80,
            "outer_step": 2,
            "inner_step": 0,
            "stage_history": [],
        }
        state = {
            "best_solution": "def old(): pass",
            "best_score": 0.85,
        }
        updates = alg2_result_to_system(result, state)
        assert "best_solution" not in updates
        assert updates["current_solution"] == "def worse(): pass"

    def test_s3_14_stage_history_propagated(self):
        from src.mle_star.graph import alg2_result_to_system

        result = {
            "improved_solution": "sol",
            "improved_score": 0.88,
            "outer_step": 1,
            "inner_step": 0,
            "stage_history": [{"stage": "A4__generate_ablation", "status": "complete"}],
        }
        state = {"best_solution": "old", "best_score": 0.85}
        updates = alg2_result_to_system(result, state)
        assert "stage_history" in updates


# ── S3-15, S3-16: Regression — previous stages still work ──────────────


class TestRegression:
    """S3-15, S3-16: Previous stages still pass."""

    def test_s3_15_algorithm1_e2e(self, e2e_result):
        from src.mle_star.algorithms.algorithm_1 import run

        assert e2e_result.get("status") == "done"
        assert len(e2e_result.get("candidates_pool", [])) > 0

    def test_s3_16_search_routes_to_parallel_pipeline(self):
        from src.mle_star.supervisor import supervisor_decide

        state = {"phase": "search", "full_cycles": 0}
        assert supervisor_decide(state) == "parallel_pipeline_node"

    def test_main_graph_still_compiles(self):
        from src.mle_star.graph import get_mle_star_graph

        graph = get_mle_star_graph()
        assert graph is not None

    def test_full_run_with_alg2(self):
        from src.mle_star.graph import run

        run_dir = tempfile.mkdtemp(prefix="mle_star_s3_full_")
        result = run(
            initial_state={
                "task_desc": "Stage 3 integration test",
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
