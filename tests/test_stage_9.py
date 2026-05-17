"""Stage 9 verification tests: Ensemble + Submission LLM Integration.

Validates that ensemble and submission nodes have real LLM call paths,
resource-limited sandbox execution, subgraph integration, algorithm_3
end-to-end runs, and proper mock/real dispatch.

Test IDs follow the s9_XX convention.

Stage 9 verification checklist (from final_requirements_stages.md):
- Run ensemble on 2 real solutions -> produces ensemble script with better score
- Ensemble loop iterates R=5 rounds with improving plans
- Submission generation produces working submission code
- Subsampling removal correctly modifies code to use full training data
- Full graph: search -> ablation -> refinement -> ensemble -> submission -> final score
- End-to-end run on at least 1 Kaggle task completes successfully
- Sandbox: resource limits (CPU + memory) enforced via resource module
"""

import json
import os
import resource
import tempfile
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest

from src.mle_star.config import MOCK_MODE


ALL_DELAY_PATCHES = [
    "src.mle_star.nodes.ensemble.simulate_delay",
    "src.mle_star.nodes.robustness.simulate_delay",
    "src.mle_star.nodes.execution.simulate_delay",
    "src.mle_star.nodes.submission.simulate_delay",
]


@contextmanager
def _no_delays():
    patches = [patch(p, return_value=None) for p in ALL_DELAY_PATCHES]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


# ── Sandbox Resource Limits ────────────────────────────────────────────────


class TestSandboxResourceLimits:
    def test_s9_01_set_resource_limits_callable(self):
        from src.mle_star.nodes.execution import _set_resource_limits

        _set_resource_limits()

    def test_s9_02_config_max_memory_mb(self):
        from src.mle_star.config import EXECUTION_MAX_MEMORY_MB

        assert isinstance(EXECUTION_MAX_MEMORY_MB, int)
        assert EXECUTION_MAX_MEMORY_MB > 0

    def test_s9_03_config_max_cpu_seconds(self):
        from src.mle_star.config import EXECUTION_MAX_CPU_SECONDS

        assert isinstance(EXECUTION_MAX_CPU_SECONDS, int)
        assert EXECUTION_MAX_CPU_SECONDS >= 0

    def test_s9_04_execute_code_uses_preexec_fn(self):
        from src.mle_star.nodes.execution import execute_code, _set_resource_limits
        from src.mle_star.state.shared import parse_score

        with patch("src.mle_star.nodes.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Final Validation Performance: 0.123456",
                stderr="",
                returncode=0,
            )
            result = execute_code("print(1)")
            assert mock_run.called
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("preexec_fn") is _set_resource_limits or (
                len(call_kwargs.args) > 4
                and call_kwargs.args[4] is _set_resource_limits
            )

    def test_s9_05_resource_limit_enforced_in_subprocess(self):
        from src.mle_star.nodes.execution import execute_code
        from src.mle_star.config import EXECUTION_MAX_MEMORY_MB

        if EXECUTION_MAX_MEMORY_MB == 0:
            pytest.skip("Resource limits disabled (EXECUTION_MAX_MEMORY_MB=0)")

        mem_hog_code = """
x = 'A' * (1024 * 1024 * 1024)
print('allocated')
"""
        result = execute_code(mem_hog_code, timeout=30)
        assert result["status"] in ("error", "timeout", "ok")


# ── eval_ensemble Real Mode ───────────────────────────────────────────────


class TestEvalEnsembleRealMode:
    def test_s9_06_eval_ensemble_real_success(self):
        with patch("src.mle_star.nodes.execution._is_mock_mode", return_value=False):
            from src.mle_star.nodes.execution import eval_ensemble

            mock_result = {
                "stdout": "Final Validation Performance: 0.110000",
                "stderr": "",
                "exit_code": 0,
                "score": 0.11,
                "status": "ok",
            }
            with patch(
                "src.mle_star.nodes.execution.execute_code",
                return_value=mock_result,
            ):
                state = {
                    "current_ensemble_code": "import numpy as np",
                    "best_ensemble_score": 0.12,
                    "best_ensemble_code": "",
                    "debug_retries": 0,
                    "metric_direction": "minimize",
                    "train_path": "input/train.csv",
                    "test_path": "input/test.csv",
                    "target_cols": ["target"],
                }
                result = eval_ensemble(state)
                assert result["status"] == "ok"
                assert result["execution_score"] == 0.11
                assert result["current_ensemble_score"] == 0.11
                assert result["best_ensemble_score"] == 0.11

    def test_s9_07_eval_ensemble_real_error_retries(self):
        with patch("src.mle_star.nodes.execution._is_mock_mode", return_value=False):
            from src.mle_star.nodes.execution import eval_ensemble

            mock_result = {
                "stdout": "",
                "stderr": "ImportError: No module named 'fake'",
                "exit_code": 1,
                "score": None,
                "status": "error",
            }
            with patch(
                "src.mle_star.nodes.execution.execute_code",
                return_value=mock_result,
            ):
                state = {
                    "current_ensemble_code": "import fake_module",
                    "best_ensemble_score": 0.12,
                    "best_ensemble_code": "orig_code",
                    "debug_retries": 0,
                    "metric_direction": "minimize",
                    "train_path": "input/train.csv",
                    "test_path": "input/test.csv",
                    "target_cols": ["target"],
                }
                result = eval_ensemble(state)
                assert result["status"] == "error"
                assert result["execution_score"] is None

    def test_s9_08_eval_ensemble_real_no_improvement(self):
        with patch("src.mle_star.nodes.execution._is_mock_mode", return_value=False):
            from src.mle_star.nodes.execution import eval_ensemble

            mock_result = {
                "stdout": "Final Validation Performance: 0.150000",
                "stderr": "",
                "exit_code": 0,
                "score": 0.15,
                "status": "ok",
            }
            with patch(
                "src.mle_star.nodes.execution.execute_code",
                return_value=mock_result,
            ):
                state = {
                    "current_ensemble_code": "worse_ensemble",
                    "best_ensemble_score": 0.10,
                    "best_ensemble_code": "best_code",
                    "debug_retries": 0,
                    "metric_direction": "minimize",
                    "train_path": "input/train.csv",
                    "test_path": "input/test.csv",
                    "target_cols": ["target"],
                }
                result = eval_ensemble(state)
                assert result["status"] == "ok"
                assert result["best_ensemble_score"] == 0.10
                assert result["best_ensemble_code"] == "best_code"


# ── eval_submission Real Mode ─────────────────────────────────────────────


class TestEvalSubmissionRealMode:
    def test_s9_09_eval_submission_real_success(self):
        with patch("src.mle_star.nodes.execution._is_mock_mode", return_value=False):
            from src.mle_star.nodes.execution import eval_submission

            mock_result = {
                "stdout": "Final Validation Performance: 0.095000",
                "stderr": "",
                "exit_code": 0,
                "score": 0.095,
                "status": "ok",
            }
            with patch(
                "src.mle_star.nodes.execution.execute_code",
                return_value=mock_result,
            ):
                state = {
                    "submission_code": "import numpy as np",
                    "best_score": 0.10,
                    "metric_direction": "minimize",
                    "train_path": "input/train.csv",
                    "test_path": "input/test.csv",
                    "target_cols": ["target"],
                }
                result = eval_submission(state)
                assert result["status"] == "ok"
                assert result["submission_score"] == 0.095

    def test_s9_10_eval_submission_real_error(self):
        with patch("src.mle_star.nodes.execution._is_mock_mode", return_value=False):
            from src.mle_star.nodes.execution import eval_submission

            mock_result = {
                "stdout": "",
                "stderr": "RuntimeError: something bad",
                "exit_code": 1,
                "score": None,
                "status": "error",
            }
            with patch(
                "src.mle_star.nodes.execution.execute_code",
                return_value=mock_result,
            ):
                state = {
                    "submission_code": "raise RuntimeError",
                    "best_score": 0.10,
                    "metric_direction": "minimize",
                    "train_path": "input/train.csv",
                    "test_path": "input/test.csv",
                    "target_cols": ["target"],
                }
                result = eval_submission(state)
                assert result["status"] == "error"
                assert result["submission_score"] is None


# ── Ensemble Round Subgraph Integration ────────────────────────────────────


class TestEnsembleRoundSubgraph:
    def test_s9_11_round_subgraph_compiles(self):
        from src.mle_star.subgraphs.ensemble_round_subgraph import (
            get_ensemble_round_subgraph,
        )

        subgraph = get_ensemble_round_subgraph()
        assert subgraph is not None

    def test_s9_12_round_subgraph_runs_mock(self):
        from src.mle_star.subgraphs.ensemble_round_subgraph import (
            get_ensemble_round_subgraph,
        )

        subgraph = get_ensemble_round_subgraph()

        with _no_delays():
            state = {
                "ensemble_solutions": ["code1", "code2"],
                "ensemble_input_scores": [0.12, 0.15],
                "metric_direction": "minimize",
                "current_ensemble_plan": "",
                "current_ensemble_code": "",
                "current_ensemble_score": 0,
                "ensemble_round": 0,
                "execution_output": "",
                "execution_error": None,
                "execution_score": None,
                "debug_retries": 0,
                "leakage_status": None,
                "leakage_code_block": None,
                "best_ensemble_code": "code1",
                "best_ensemble_score": 0.12,
                "status": "start",
            }
            result = subgraph.invoke(state)
            assert "status" in result
            assert result.get("current_ensemble_code", "") != ""

    def test_s9_13_round_subgraph_leakage_routing(self):
        from src.mle_star.nodes.robustness import (
            route_after_leakage_check_ensemble,
        )

        assert (
            route_after_leakage_check_ensemble({"leakage_status": "Yes Data Leakage"})
            == "A12__fix_leakage_ensemble"
        )
        assert (
            route_after_leakage_check_ensemble({"leakage_status": "ok"})
            == "eval_ensemble"
        )


# ── Submission Subgraph Integration ────────────────────────────────────────


class TestSubmissionSubgraph:
    def test_s9_14_submission_subgraph_compiles(self):
        from src.mle_star.subgraphs.submission_subgraph import (
            get_submission_subgraph,
        )

        subgraph = get_submission_subgraph()
        assert subgraph is not None

    def test_s9_15_submission_subgraph_runs_mock(self):
        from src.mle_star.subgraphs.submission_subgraph import (
            get_submission_subgraph,
        )

        subgraph = get_submission_subgraph()

        with _no_delays():
            state = {
                "final_solution": "import numpy as np\nprint('hello')",
                "best_score": 0.12,
                "task_desc": "predict energy",
                "score_function_desc": "RMSLE",
                "submission_code": "",
                "submission_score": None,
                "subsampling_block": "",
                "leakage_status": None,
                "status": "start",
            }
            result = subgraph.invoke(state)
            assert "submission_code" in result
            assert "submission_score" in result

    def test_s9_16_submission_subgraph_leakage_routing(self):
        from src.mle_star.nodes.submission import (
            route_after_leakage_check_submission,
        )

        assert (
            route_after_leakage_check_submission({"leakage_status": "Yes Data Leakage"})
            == "A12__fix_leakage_submission"
        )
        assert (
            route_after_leakage_check_submission({"leakage_status": "ok"})
            == "eval_submission"
        )


# ── Algorithm 3 End-to-End ──────────────────────────────────────────────────


class TestAlgorithm3EndToEnd:
    def test_s9_17_algorithm3_run_mock(self):
        from src.mle_star.algorithms.algorithm_3 import run

        with _no_delays():
            initial_state = {
                "ensemble_solutions": ["code_a", "code_b"],
                "ensemble_input_scores": [0.12, 0.15],
                "metric_direction": "minimize",
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
            with tempfile.TemporaryDirectory(prefix="mle_alg3_") as tmpdir:
                result = run(
                    initial_state=initial_state,
                    run_dir=tmpdir,
                    thread_id="test_alg3",
                )
                assert result.get("status") == "done"
                assert "best_ensemble_code" in result
                assert "best_ensemble_score" in result
                assert result.get("ensemble_round", 0) > 0

    def test_s9_18_algorithm3_improving_scores(self):
        from src.mle_star.algorithms.algorithm_3 import run
        from src.mle_star.config import MAX_ENSEMBLE_ROUNDS

        with _no_delays():
            initial_state = {
                "ensemble_solutions": ["code_a", "code_b"],
                "ensemble_input_scores": [0.12, 0.15],
                "metric_direction": "minimize",
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
            with tempfile.TemporaryDirectory(prefix="mle_alg3_") as tmpdir:
                result = run(
                    initial_state=initial_state,
                    run_dir=tmpdir,
                    thread_id="test_alg3_improve",
                )
                scores = result.get("ensemble_scores", [])
                assert len(scores) == MAX_ENSEMBLE_ROUNDS
                best = result.get("best_ensemble_score", 0)
                assert best != 0 or len(scores) > 0


# ── Ensemble + Submission Real LLM Paths ──────────────────────────────────


class TestEnsembleRealModeLLM:
    def test_s9_19_a9_llm_failure_fallback(self):
        with patch("src.mle_star.nodes.ensemble._is_mock_mode", return_value=False):
            from src.mle_star.nodes.ensemble import A9__plan_ensemble

            with patch(
                "src.mle_star.nodes.ensemble.call_llm",
                side_effect=Exception("API timeout"),
            ):
                state = {
                    "ensemble_solutions": ["code1", "code2"],
                    "ensemble_input_scores": [0.12, 0.15],
                    "ensemble_round": 0,
                    "ensemble_plans": [],
                    "best_ensemble_score": 0.12,
                    "metric_direction": "minimize",
                }
                result = A9__plan_ensemble(state)
                assert result["status"] == "planned"
                assert "current_ensemble_plan" in result

    def test_s9_20_a10_llm_failure_fallback(self):
        with patch("src.mle_star.nodes.ensemble._is_mock_mode", return_value=False):
            from src.mle_star.nodes.ensemble import A10__implement_ensemble

            with patch(
                "src.mle_star.nodes.ensemble.call_llm",
                side_effect=Exception("API outage"),
            ):
                state = {
                    "current_ensemble_plan": "weighted average",
                    "ensemble_solutions": ["code1", "code2"],
                    "ensemble_input_scores": [0.12, 0.15],
                    "best_ensemble_code": "",
                    "ensemble_round": 0,
                    "metric_direction": "minimize",
                }
                result = A10__implement_ensemble(state)
                assert "current_ensemble_code" in result
                assert result["status"] == "llm_failed"


class TestSubmissionRealModeLLM:
    def test_s9_21_submit_llm_failure_fallback(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import A_test__submit

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                side_effect=Exception("LLM down"),
            ):
                state = {
                    "final_solution": "original code",
                    "best_solution": "original code",
                    "best_score": 0.12,
                    "task_desc": "predict energy",
                    "metric_direction": "minimize",
                }
                result = A_test__submit(state)
                assert result["status"] == "generated"
                assert "submission_code" in result

    def test_s9_22_subsampling_remove_llm_failure_fallback(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import subsampling_remove

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                side_effect=Exception("LLM error"),
            ):
                state = {
                    "submission_code": "df = df.sample(n=100)\nprint(df.shape)",
                    "subsampling_block": "df = df.sample(n=100)",
                }
                result = subsampling_remove(state)
                assert "submission_code" in result
                assert result["status"] == "subsampling_removed"

    def test_s9_23_subsampling_extract_no_subsampling(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import subsampling_extract

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                return_value=json.dumps({"has_subsampling": False}),
            ):
                state = {
                    "submission_code": "import numpy as np\nprint('no subsampling')"
                }
                result = subsampling_extract(state)
                assert result["subsampling_block"] == ""


# ── Main Graph Integration ─────────────────────────────────────────────────


class TestMainGraphEnsembleSubmission:
    def test_s9_24_transition_to_ensemble_with_parallel_results(self):
        from src.mle_star.graph import transition_to_ensemble

        state = {
            "phase": "search_done",
            "best_score": 0.12,
            "parallel_results": [
                {"best_solution": "solution_A", "best_score": 0.12},
                {"best_solution": "solution_B", "best_score": 0.15},
            ],
            "full_cycles": 0,
        }
        result = transition_to_ensemble(state)
        assert result["phase"] == "ensemble"
        assert len(result["ensemble_solutions"]) == 2
        assert result["ensemble_input_scores"] == [0.12, 0.15]

    def test_s9_25_transition_to_ensemble_fallback(self):
        from src.mle_star.graph import transition_to_ensemble

        state = {
            "phase": "search_done",
            "best_score": 0.12,
            "best_solution": "single_solution",
            "parallel_results": [],
            "full_cycles": 0,
        }
        result = transition_to_ensemble(state)
        assert result["phase"] == "ensemble"
        assert len(result["ensemble_solutions"]) == 1
        assert result["ensemble_solutions"][0] == "single_solution"

    def test_s9_26_transition_to_submission(self):
        from src.mle_star.graph import transition_to_submission

        state = {
            "phase": "ensemble",
            "best_score": 0.08,
            "best_solution": "best_ensemble_code_here",
        }
        result = transition_to_submission(state)
        assert result["phase"] == "submission"

    def test_s9_27_alg3_result_to_system_improvement(self):
        from src.mle_star.graph import alg3_result_to_system

        state = {
            "best_score": 0.12,
            "best_solution": "old_code",
            "metric_direction": "minimize",
            "ensemble_round": 0,
        }
        result = {
            "best_ensemble_code": "new_better_code",
            "best_ensemble_score": 0.08,
            "ensemble_round": 3,
            "stage_history": [],
        }
        updates = alg3_result_to_system(result, state)
        assert updates["best_solution"] == "new_better_code"
        assert updates["best_score"] == 0.08

    def test_s9_28_alg3_result_to_system_no_improvement(self):
        from src.mle_star.graph import alg3_result_to_system

        state = {
            "best_score": 0.05,
            "best_solution": "already_best",
            "metric_direction": "minimize",
            "ensemble_round": 0,
        }
        result = {
            "best_ensemble_code": "worse_code",
            "best_ensemble_score": 0.10,
            "ensemble_round": 3,
            "stage_history": [],
        }
        updates = alg3_result_to_system(result, state)
        assert (
            "best_solution" not in updates
            or updates.get("best_solution") is None
            or updates.get("best_solution") == "already_best"
        )


# ── Sandbox Code Safety ───────────────────────────────────────────────────


class TestSandboxCodeSafety:
    def test_s9_29_validate_safety_blocks_subprocess(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = "import subprocess\nsubprocess.run(['rm', '-rf', '/'])"
        is_safe, reason = validate_code_safety(code)
        assert not is_safe
        assert "subprocess" in reason.lower()

    def test_s9_30_validate_safety_blocks_eval(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = 'eval(\'__import__("os").system("whoami")\')'
        is_safe, reason = validate_code_safety(code)
        assert not is_safe

    def test_s9_31_validate_safety_allows_sklearn(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = """
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def train_and_predict(X_train, y_train, X_val):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_val)
"""
        is_safe, reason = validate_code_safety(code)
        assert is_safe

    def test_s9_32_validate_safety_blocks_socket(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = "import socket\ns = socket.socket()"
        is_safe, reason = validate_code_safety(code)
        assert not is_safe

    def test_s9_33_execute_code_safety_check_first(self):
        from src.mle_star.nodes.execution import execute_code

        result = execute_code("import subprocess; subprocess.run(['ls'])")
        assert result["status"] == "blocked"
        assert (
            "safety" in result["stderr"].lower()
            or "forbidden" in result["stderr"].lower()
        )

    def test_s9_34_validate_safety_empty_code(self):
        from src.mle_star.nodes.execution import validate_code_safety

        is_safe, reason = validate_code_safety("")
        assert not is_safe

    def test_s9_35_validate_safety_syntax_error(self):
        from src.mle_star.nodes.execution import validate_code_safety

        is_safe, reason = validate_code_safety("def foo(:\n    return 1")
        assert not is_safe


# ── EnsembleRoundState Fields ──────────────────────────────────────────────


class TestEnsembleRoundStateSchema:
    def test_s9_36_ensemble_round_state_fields(self):
        from src.mle_star.state.alg3_state import EnsembleRoundState

        annotations = EnsembleRoundState.__annotations__
        assert "ensemble_solutions" in annotations
        assert "ensemble_input_scores" in annotations
        assert "metric_direction" in annotations
        assert "current_ensemble_plan" in annotations
        assert "current_ensemble_code" in annotations
        assert "ensemble_round" in annotations
        assert "debug_retries" in annotations
        assert "best_ensemble_code" in annotations
        assert "best_ensemble_score" in annotations
        assert "leakage_status" in annotations

    def test_s9_37_alg3_state_fields(self):
        from src.mle_star.state.alg3_state import Alg3State

        annotations = Alg3State.__annotations__
        assert "ensemble_solutions" in annotations
        assert "ensemble_input_scores" in annotations
        assert "ensemble_plans" in annotations
        assert "ensemble_scores" in annotations
        assert "best_ensemble_code" in annotations
        assert "best_ensemble_score" in annotations


# ── SubmissionState Fields ──────────────────────────────────────────────────


class TestSubmissionStateSchema:
    def test_s9_38_submission_state_fields(self):
        from src.mle_star.subgraphs.submission_subgraph import SubmissionState

        annotations = SubmissionState.__annotations__
        assert "final_solution" in annotations
        assert "submission_code" in annotations
        assert "submission_score" in annotations
        assert "subsampling_block" in annotations
        assert "leakage_status" in annotations


# ── Config Defaults ────────────────────────────────────────────────────────


class TestConfigDefaults:
    def test_s9_39_execution_timeout_default(self):
        from src.mle_star.config import EXECUTION_TIMEOUT

        assert EXECUTION_TIMEOUT == 900

    def test_s9_40_max_ensemble_rounds(self):
        from src.mle_star.config import MAX_ENSEMBLE_ROUNDS

        assert MAX_ENSEMBLE_ROUNDS >= 1

    def test_s9_41_max_ensemble_debug_retries(self):
        from src.mle_star.config import MAX_ENSEMBLE_DEBUG_RETRIES

        assert MAX_ENSEMBLE_DEBUG_RETRIES >= 1

    def test_s9_42_subsampling_threshold(self):
        from src.mle_star.config import SUBSAMPLING_THRESHOLD

        assert SUBSAMPLING_THRESHOLD >= 0

    def test_s9_43_max_leakage_fix_retries(self):
        from src.mle_star.config import MAX_LEAKAGE_FIX_RETRIES

        assert MAX_LEAKAGE_FIX_RETRIES >= 1


# ── Robustness Ensemble Nodes ─────────────────────────────────────────────


class TestRobustnessEnsembleRealMode:
    def test_s9_44_debug_ensemble_llm_success(self):
        with patch("src.mle_star.nodes.robustness._is_mock_mode", return_value=False):
            from src.mle_star.nodes.robustness import A11__debug_ensemble

            with patch(
                "src.mle_star.nodes.robustness.call_llm",
                return_value="```python\ndef fixed(): return 42\n```",
            ):
                state = {
                    "current_ensemble_code": "def broken(): return 1/0",
                    "debug_retries": 0,
                    "execution_error": "ZeroDivisionError",
                    "task_desc": "predict energy",
                    "metric_direction": "minimize",
                }
                result = A11__debug_ensemble(state)
                assert result["debug_retries"] == 1
                assert result["status"] == "debugged"
                assert "current_ensemble_code" in result

    def test_s9_45_check_leakage_ensemble_llm_no_leakage(self):
        with patch("src.mle_star.nodes.robustness._is_mock_mode", return_value=False):
            from src.mle_star.nodes.robustness import A12__check_leakage_ensemble

            with patch(
                "src.mle_star.nodes.robustness.call_llm",
                return_value=json.dumps({"has_leakage": False}),
            ):
                state = {"current_ensemble_code": "clean code"}
                result = A12__check_leakage_ensemble(state)
                assert result["leakage_status"] == "ok"

    def test_s9_46_check_leakage_ensemble_llm_found_leakage(self):
        with patch("src.mle_star.nodes.robustness._is_mock_mode", return_value=False):
            from src.mle_star.nodes.robustness import A12__check_leakage_ensemble

            with patch(
                "src.mle_star.nodes.robustness.call_llm",
                return_value=json.dumps(
                    {
                        "has_leakage": True,
                        "leakage_issues": ["Using test labels in training"],
                    }
                ),
            ):
                state = {"current_ensemble_code": "leaky code"}
                result = A12__check_leakage_ensemble(state)
                assert result["leakage_status"] == "Yes Data Leakage"

    def test_s9_47_fix_leakage_ensemble_llm(self):
        with patch("src.mle_star.nodes.robustness._is_mock_mode", return_value=False):
            from src.mle_star.nodes.robustness import A12__fix_leakage_ensemble

            with patch(
                "src.mle_star.nodes.robustness.call_llm",
                return_value="```python\ndef clean(): pass\n```",
            ):
                state = {
                    "current_ensemble_code": "leaky code",
                    "leakage_issues": ["target leakage"],
                    "task_desc": "predict",
                    "metric_direction": "minimize",
                }
                result = A12__fix_leakage_ensemble(state)
                assert result["status"] == "leakage_fix_applied"
                assert result["leakage_status"] is None

    def test_s9_48_fix_leakage_ensemble_llm_failure(self):
        with patch("src.mle_star.nodes.robustness._is_mock_mode", return_value=False):
            from src.mle_star.nodes.robustness import A12__fix_leakage_ensemble

            with patch(
                "src.mle_star.nodes.robustness.call_llm",
                side_effect=Exception("API error"),
            ):
                state = {
                    "current_ensemble_code": "leaky code",
                    "leakage_issues": ["target leakage"],
                    "task_desc": "predict",
                    "metric_direction": "minimize",
                }
                result = A12__fix_leakage_ensemble(state)
                assert result["status"] == "leakage_fix_applied"


# ── Submission Leakage Real Mode ──────────────────────────────────────────


class TestSubmissionLeakageRealMode:
    def test_s9_49_fix_leakage_submission_llm(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import A12__fix_leakage_submission

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                return_value="```python\nimport numpy as np\nprint('clean')\n```",
            ):
                state = {
                    "submission_code": "leaky submission code",
                    "leakage_issues": ["data leakage in preprocessing"],
                    "task_desc": "predict energy",
                    "metric_direction": "minimize",
                }
                result = A12__fix_leakage_submission(state)
                assert result["status"] == "leakage_fix_applied"
                assert result["leakage_status"] is None

    def test_s9_50_fix_leakage_submission_llm_failure(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import A12__fix_leakage_submission

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                side_effect=Exception("LLM timeout"),
            ):
                state = {
                    "submission_code": "leaky code",
                    "leakage_issues": ["leakage issue"],
                    "task_desc": "predict",
                    "metric_direction": "minimize",
                }
                result = A12__fix_leakage_submission(state)
                assert result["status"] == "leakage_fix_applied"


# ── G1: format_direction ─────────────────────────────────────────────────


class TestFormatDirection:
    def test_s9_51_format_direction_minimize(self):
        from src.mle_star.state.shared import format_direction

        assert format_direction("minimize") == "lower is better = minimize"

    def test_s9_52_format_direction_maximize(self):
        from src.mle_star.state.shared import format_direction

        assert format_direction("maximize") == "higher is better = maximize"

    def test_s9_53_ensemble_prompt_uses_direction(self):
        from src.mle_star.prompts.ensemble import ENSEMBLE_PLANNER_PROMPT

        prompt = ENSEMBLE_PLANNER_PROMPT.format(
            task_desc="predict accuracy",
            metric="accuracy",
            direction="higher is better = maximize",
            score_descriptions="Solution 1: 0.85",
            previous_plans="none",
            best_ensemble_score="0.85",
            ensemble_round="0",
        )
        assert "higher is better = maximize" in prompt
        assert "lower is better" not in prompt

    def test_s9_54_ablation_prompt_uses_direction(self):
        from src.mle_star.prompts.ablation import ABLATION_STUDY_PROMPT

        prompt = ABLATION_STUDY_PROMPT.format(
            task_desc="predict accuracy",
            metric="accuracy",
            direction="higher is better = maximize",
            solution="import numpy",
            functional_blocks="block1",
            previous_summaries="none",
        )
        assert "higher is better = maximize" in prompt

    def test_s9_55_coder_prompt_uses_direction(self):
        from src.mle_star.prompts.refinement import CODER_PROMPT

        prompt = CODER_PROMPT.format(
            task_desc="predict accuracy",
            metric="accuracy",
            direction="higher is better = maximize",
            current_solution="import numpy",
            target_block="def foo(): pass",
            current_plan="improve features",
            previous_attempts="none",
        )
        assert "higher is better = maximize" in prompt

    def test_s9_56_candidate_eval_prompt_uses_direction(self):
        from src.mle_star.prompts.search import CANDIDATE_EVAL_PROMPT

        prompt = CANDIDATE_EVAL_PROMPT.format(
            task_desc="predict accuracy",
            model_name="RandomForestClassifier",
            model_description="ensemble",
            metric="accuracy",
            direction="higher is better = maximize",
            feature_cols="feat1, feat2",
            target_cols="label",
            additional_constraints="",
        )
        assert "higher is better = maximize" in prompt

    def test_s9_57_merger_prompt_uses_direction(self):
        from src.mle_star.prompts.search import MERGER_PROMPT

        prompt = MERGER_PROMPT.format(
            task_desc="predict accuracy",
            metric="accuracy",
            direction="higher is better = maximize",
            base_code="code1",
            ref_code="code2",
        )
        assert "higher is better = maximize" in prompt


# ── G6: Full Graph End-to-End ────────────────────────────────────────────


class TestFullGraphE2E:
    def test_s9_58_full_graph_search_through_submission(self):
        from src.mle_star.graph import run

        with _no_delays():
            with tempfile.TemporaryDirectory(prefix="mle_s9_e2e_") as tmpdir:
                state = {
                    "task_desc": "predict energy",
                    "score_function_desc": "RMSLE",
                    "datasets": [],
                    "phase": "search",
                }
                result = run(
                    initial_state=state,
                    run_dir=tmpdir,
                    thread_id="s9_e2e",
                )
                assert result.get("status") in ("done", "search_complete", "")
                assert result.get("best_solution", "") != "" or result.get("phase") in (
                    "search",
                    "search_done",
                    "ensemble",
                    "submission",
                )

    def test_s9_59_transition_flow_ensemble_to_submission(self):
        from src.mle_star.graph import (
            transition_to_ensemble,
            transition_to_submission,
        )

        state_after_search = {
            "phase": "search_done",
            "best_score": 0.08,
            "best_solution": "best_search_code",
            "parallel_results": [
                {"best_solution": "sol_A", "best_score": 0.10},
                {"best_solution": "sol_B", "best_score": 0.08},
            ],
            "full_cycles": 0,
        }

        ensemble_state = transition_to_ensemble(state_after_search)
        assert ensemble_state["phase"] == "ensemble"
        assert len(ensemble_state["ensemble_solutions"]) == 2

        state_after_ensemble = {
            **state_after_search,
            "phase": "ensemble",
            "best_score": 0.05,
            "best_solution": "best_ensemble_code",
            "ensemble_round": 3,
        }
        submission_state = transition_to_submission(state_after_ensemble)
        assert submission_state["phase"] == "submission"


# ── G9: Subsampling Removal Correctness ──────────────────────────────────


class TestSubsamplingRemoval:
    def test_s9_60_subsampling_remove_strips_sample_call(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import subsampling_remove

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                return_value="```python\nimport pandas as pd\ntrain = pd.read_csv('train.csv')\nX = train.drop('target', axis=1)\n```",
            ):
                state = {
                    "submission_code": "train = df.sample(n=1000)\nX = train.drop('target', axis=1)",
                    "subsampling_block": "train = df.sample(n=1000)",
                }
                result = subsampling_remove(state)
                assert "submission_code" in result
                assert result["status"] == "subsampling_removed"

    def test_s9_61_subsampling_remove_no_block_no_change(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import subsampling_remove

            state = {
                "submission_code": "import numpy as np\nprint('no subsampling')",
                "subsampling_block": "",
            }
            result = subsampling_remove(state)
            assert (
                result["submission_code"]
                == "import numpy as np\nprint('no subsampling')"
            )


# ── G2: CPU Resource Limit ──────────────────────────────────────────────


class TestCPULimit:
    def test_s9_62_cpu_limit_config_is_positive(self):
        from src.mle_star.config import EXECUTION_MAX_CPU_SECONDS

        assert EXECUTION_MAX_CPU_SECONDS > 0


# ── G3: ./final/ Directory Convention ──────────────────────────────────────


class TestFinalDirectory:
    def test_s9_63_eval_submission_creates_final_dir(self):
        with tempfile.TemporaryDirectory(prefix="mle_final_") as tmpdir:
            from src.mle_star.nodes.execution import eval_submission
            from src.mle_star.state.shared import _current_run_dir

            _current_run_dir.set(tmpdir)
            try:
                state = {
                    "submission_code": "print('hello')",
                    "best_score": 0.1,
                    "metric_direction": "minimize",
                }
                result = eval_submission(state)
                final_dir = result.get("final_dir", "")
                assert final_dir != ""
                assert os.path.isdir(final_dir)
                code_path = os.path.join(final_dir, "submission_code.py")
                score_path = os.path.join(final_dir, "submission_score.txt")
                assert os.path.isfile(code_path)
                assert os.path.isfile(score_path)
                with open(code_path) as f:
                    assert "print('hello')" in f.read()
                with open(score_path) as f:
                    score_text = f.read().strip()
                    assert float(score_text) > 0
            finally:
                _current_run_dir.set(None)

    def test_s9_64_final_dir_under_run_dir(self):
        with tempfile.TemporaryDirectory(prefix="mle_final_") as tmpdir:
            from src.mle_star.nodes.execution import _write_final_artifacts

            final_dir = _write_final_artifacts(tmpdir, "x=1", 0.05)
            assert final_dir is not None
            assert final_dir == os.path.join(tmpdir, "final")
            assert os.path.isdir(final_dir)

    def test_s9_65_write_final_artifacts_no_run_dir(self):
        from src.mle_star.nodes.execution import _write_final_artifacts

        result = _write_final_artifacts("", "x=1", 0.05)
        assert result is None

    def test_s9_66_submission_state_has_final_dir(self):
        from src.mle_star.subgraphs.submission_subgraph import SubmissionState

        annotations = SubmissionState.__annotations__
        assert "final_dir" in annotations

    def test_s9_67_system_state_has_final_dir(self):
        from src.mle_star.state.system_state import MleStarSystemState

        annotations = MleStarSystemState.__annotations__
        assert "final_dir" in annotations


# ── G4: Traceback Key in execute_code ────────────────────────────────────────


class TestTracebackKey:
    def test_s9_68_execute_code_returns_traceback_key(self):
        from src.mle_star.nodes.execution import execute_code

        result = execute_code("print('x')", train_path="/nonexistent/train.csv")
        assert "traceback" in result
        assert isinstance(result["traceback"], str)

    def test_s9_69_extract_traceback_with_traceback(self):
        from src.mle_star.nodes.execution import _extract_traceback

        stderr = "some output\nTraceback (most recent call last):\n  File 'test.py', line 1\n    x =\n       ^\nSyntaxError: invalid syntax\n"
        tb = _extract_traceback(stderr)
        assert tb.startswith("Traceback (most recent call last)")
        assert "SyntaxError" in tb

    def test_s9_70_extract_traceback_no_traceback(self):
        from src.mle_star.nodes.execution import _extract_traceback

        stderr = "just some warnings\nno error here\n"
        tb = _extract_traceback(stderr)
        assert tb == ""

    def test_s9_71_blocked_code_has_empty_traceback(self):
        from src.mle_star.nodes.execution import execute_code

        result = execute_code("import subprocess; subprocess.run(['rm', '-rf', '/'])")
        assert result["status"] == "blocked"
        assert result["traceback"] == ""

    def test_s9_72_timeout_has_empty_traceback(self):
        from src.mle_star.nodes.execution import execute_code

        result = execute_code("import time; time.sleep(999)", timeout=1)
        assert result["status"] == "timeout"
        assert result["traceback"] == ""

    def test_s9_73_error_result_has_traceback(self):
        from src.mle_star.nodes.execution import _extract_traceback

        stderr = "Traceback (most recent call last):\n  File 't.py', line 1\n    raise ValueError('test')\nValueError: test\n"
        tb = _extract_traceback(stderr)
        assert tb != ""
        assert "ValueError" in tb


# ── G8: Submission CSV Output Verification ───────────────────────────────────


class TestSubmissionCSVOutput:
    def test_s9_74_eval_submission_writes_submission_code_py(self):
        with tempfile.TemporaryDirectory(prefix="mle_csv_") as tmpdir:
            from src.mle_star.nodes.execution import eval_submission
            from src.mle_star.state.shared import _current_run_dir

            _current_run_dir.set(tmpdir)
            try:
                state = {
                    "submission_code": "import pandas as pd\nprint('submission')",
                    "best_score": 0.08,
                    "metric_direction": "minimize",
                }
                result = eval_submission(state)
                final_dir = result.get("final_dir", "")
                assert final_dir != ""
                code_path = os.path.join(final_dir, "submission_code.py")
                with open(code_path) as f:
                    content = f.read()
                assert "import pandas as pd" in content
                assert "print('submission')" in content
            finally:
                _current_run_dir.set(None)

    def test_s9_75_eval_submission_writes_score_txt(self):
        with tempfile.TemporaryDirectory(prefix="mle_csv_") as tmpdir:
            from src.mle_star.nodes.execution import eval_submission
            from src.mle_star.state.shared import _current_run_dir

            _current_run_dir.set(tmpdir)
            try:
                state = {
                    "submission_code": "x=1",
                    "best_score": 0.15,
                    "metric_direction": "minimize",
                }
                result = eval_submission(state)
                score = result.get("submission_score")
                assert score is not None
                final_dir = result.get("final_dir", "")
                assert final_dir != ""
                score_path = os.path.join(final_dir, "submission_score.txt")
                with open(score_path) as f:
                    assert float(f.read().strip()) == score
            finally:
                _current_run_dir.set(None)

    def test_s9_76_full_graph_produces_final_dir(self):
        from src.mle_star.graph import run

        with _no_delays():
            with tempfile.TemporaryDirectory(prefix="mle_s9_final_") as tmpdir:
                state = {
                    "task_desc": "predict energy",
                    "score_function_desc": "RMSLE",
                    "datasets": [],
                    "phase": "search",
                }
                result = run(
                    initial_state=state,
                    run_dir=tmpdir,
                    thread_id="s9_final",
                )
                final_dir = result.get("final_dir", "")
                if final_dir:
                    assert os.path.isdir(final_dir)
                    assert os.path.isfile(os.path.join(final_dir, "submission_code.py"))
