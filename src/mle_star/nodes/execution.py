"""Execution nodes: eval_candidate, eval_merge, eval_refinement, eval_ensemble, eval_submission.

Provides subprocess-based code execution with sandbox validation for the
search phase, and mock implementations for ensemble/submission (Stages 8-9).

Real execution uses subprocess.run() with timeout, AST-based safety validation,
and resource limits (CPU time + memory via the resource module).
Mock mode (MLE_MOCK_MODE) returns mock scores for testing without real execution.

Stage 10: Added Docker sandbox support. When DOCKER_SANDBOX_ENABLED is true
and Docker is available, code executes in ephemeral containers with:
- Network disabled (none)
- Resource limits (CPU, RAM)
- Read-only mount for input/ data
- Read-write mount for final/ output
- Automatic container cleanup

Falls back to subprocess execution if Docker is unavailable.
"""

import ast
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Dict, List, Optional, Tuple

from src.mle_star.config import (
    EXECUTION_TIMEOUT,
    EXECUTION_MAX_MEMORY_MB,
    EXECUTION_MAX_CPU_SECONDS,
    MAX_DEBUG_RETRIES,
    MAX_ENSEMBLE_DEBUG_RETRIES,
    MOCK_MODE,
    DOCKER_SANDBOX_ENABLED,
    DOCKER_IMAGE,
    DOCKER_CPU_LIMIT,
    DOCKER_MEMORY_LIMIT_MB,
)
from src.mle_star.state.shared import (
    traceable,
    simulate_delay,
    log_node_event,
    random_score,
    random_pass,
    normalize_score,
    parse_score,
    get_run_dir,
)


def _extract_traceback(stderr: str) -> str:
    """Extract the formatted traceback from stderr output.

    Returns the portion of stderr starting from 'Traceback (most recent call
    last)' through the last line of the exception. Returns empty string if no
    traceback is found.
    """
    marker = "Traceback (most recent call last)"
    idx = stderr.find(marker)
    if idx == -1:
        return ""
    return stderr[idx:]


def _log_execution_artifact(
    user_code: str,
    wrapped_code: str,
    result: Dict,
    timeout: int,
    work_dir: str | None = None,
):
    """Write execution code and output to run_dir/exec_log/ for debugging."""
    run_dir = get_run_dir()
    exec_log_dir = os.path.join(run_dir, "exec_log")
    os.makedirs(exec_log_dir, exist_ok=True)

    ts = int(time.time() * 1000)
    status = result.get("status", "unknown")
    exit_code = result.get("exit_code", -1)
    score = result.get("score")

    prefix = f"{ts}_{status}_exit{exit_code}"

    try:
        with open(os.path.join(exec_log_dir, f"{prefix}_user_code.py"), "w") as f:
            f.write(user_code if user_code else "# (empty)")

        with open(os.path.join(exec_log_dir, f"{prefix}_wrapped.py"), "w") as f:
            f.write(wrapped_code if wrapped_code else "# (empty)")

        with open(os.path.join(exec_log_dir, f"{prefix}_result.json"), "w") as f:
            json.dump(
                {
                    "status": status,
                    "exit_code": exit_code,
                    "score": score,
                    "timeout": timeout,
                    "stdout_len": len(result.get("stdout", "")),
                    "stderr_len": len(result.get("stderr", "")),
                    "stdout_preview": result.get("stdout", "")[:2000],
                    "stderr_preview": result.get("stderr", "")[:2000],
                    "traceback": result.get("traceback", ""),
                    "work_dir": work_dir,
                },
                f,
                indent=2,
                default=str,
            )
    except Exception as e:
        log_node_event("_log_execution_artifact", "error", {"error": str(e)[:200]})


def _set_resource_limits():
    """Pre-exec hook for subprocess: set CPU time and memory resource limits.

    Called via subprocess.run(preexec_fn=) before the user code executes.
    Falls back gracefully on platforms where resource limits are unavailable.
    """
    try:
        cpu_limit = EXECUTION_MAX_CPU_SECONDS
        if cpu_limit > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except (ValueError, OSError):
        pass

    try:
        mem_bytes = EXECUTION_MAX_MEMORY_MB * 1024 * 1024
        if mem_bytes > 0:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass


def _docker_available() -> bool:
    """Check if Docker is available and the daemon is running."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class DockerSandbox:
    """Execute Python code in an ephemeral Docker container.

    Creates a container with:
    - Network disabled (none mode)
    - CPU and memory limits
    - Read-only mount for input/ data
    - Read-write mount for final/ output
    - Automatic cleanup on exit

    Usage:
        with DockerSandbox(work_dir="/path/to/run") as sandbox:
            result = sandbox.run(code, timeout=600)
    """

    def __init__(
        self,
        work_dir: str,
        image: str | None = None,
        cpu_limit: int | None = None,
        memory_limit_mb: int | None = None,
    ):
        self.work_dir = work_dir
        self.image = image or DOCKER_IMAGE
        self.cpu_limit = cpu_limit or DOCKER_CPU_LIMIT
        self.memory_limit_mb = memory_limit_mb or DOCKER_MEMORY_LIMIT_MB
        self._client = None
        self._container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def _get_client(self):
        if self._client is None:
            import docker

            self._client = docker.from_env()
        return self._client

    def run(self, code: str, timeout: int = EXECUTION_TIMEOUT) -> Dict:
        """Execute code in a Docker container.

        Args:
            code: Python code string to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            Dict with keys: stdout, stderr, exit_code, score, status, traceback.
        """
        import docker

        client = self._get_client()

        script_path = os.path.join(self.work_dir, "_sandbox_script.py")
        with open(script_path, "w") as f:
            f.write(code)

        input_dir = os.path.join(self.work_dir, "input")
        final_dir = os.path.join(self.work_dir, "final")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        volumes = {
            os.path.abspath(input_dir): {"bind": "/workspace/input", "mode": "ro"},
            os.path.abspath(final_dir): {"bind": "/workspace/final", "mode": "rw"},
            os.path.abspath(script_path): {
                "bind": "/workspace/script.py",
                "mode": "ro",
            },
        }

        mem_limit = f"{self.memory_limit_mb}m"

        try:
            self._container = client.containers.run(
                image=self.image,
                command=["python", "/workspace/script.py"],
                volumes=volumes,
                network_mode="none",
                mem_limit=mem_limit,
                nano_cpus=int(self.cpu_limit * 1e9),
                working_dir="/workspace",
                detach=True,
                stdout=True,
                stderr=True,
            )

            try:
                result = self._container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                try:
                    self._container.kill()
                except Exception:
                    pass
                log_node_event("DockerSandbox", "timeout", {"timeout": timeout})
                return {
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout} seconds",
                    "exit_code": -1,
                    "score": None,
                    "status": "timeout",
                    "traceback": "",
                }

            stdout = self._container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = self._container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )

            score = parse_score(stdout)

            log_node_event(
                "DockerSandbox",
                "result",
                {
                    "exit_code": exit_code,
                    "score": score,
                    "stdout_len": len(stdout),
                    "stderr_len": len(stderr),
                },
            )

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "score": score,
                "status": "ok" if exit_code == 0 and score is not None else "error",
                "traceback": _extract_traceback(stderr),
            }

        except docker.errors.ImageNotFound:
            log_node_event("DockerSandbox", "image_not_found", {"image": self.image})
            return {
                "stdout": "",
                "stderr": f"Docker image not found: {self.image}",
                "exit_code": -1,
                "score": None,
                "status": "error",
                "traceback": "",
            }
        except Exception as e:
            log_node_event("DockerSandbox", "error", {"error": str(e)[:200]})
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "score": None,
                "status": "error",
                "traceback": "",
            }
        finally:
            self.cleanup()
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def cleanup(self):
        """Remove the container if it exists."""
        if self._container is not None:
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None


def _execute_in_docker(
    code: str, timeout: int = EXECUTION_TIMEOUT, work_dir: str | None = None
) -> Dict:
    """Execute code in a Docker sandbox.

    Falls back to subprocess execution if Docker is unavailable.

    Args:
        code: Python code string to execute.
        timeout: Maximum execution time in seconds.
        work_dir: Working directory for mounts.

    Returns:
        Dict with keys: stdout, stderr, exit_code, score, status, traceback.
    """
    if work_dir is None:
        work_dir = os.getcwd()

    is_safe, reason = validate_code_safety(code)
    if not is_safe:
        log_node_event(
            "_execute_in_docker",
            "safety_check",
            {"status": "blocked", "reason": reason},
        )
        return {
            "stdout": "",
            "stderr": f"Code safety check failed: {reason}",
            "exit_code": -1,
            "score": None,
            "status": "blocked",
            "traceback": "",
        }

    if not _docker_available():
        log_node_event(
            "_execute_in_docker",
            "fallback",
            {"reason": "Docker unavailable, using subprocess"},
        )
        wrapped_code = EXECUTION_TEMPLATE.format(
            train_path="input/train.csv",
            test_path="input/test.csv",
            target_cols=["formation_energy_ev_natom", "bandgap_energy_ev"],
            user_code=code,
        )
        return _execute_subprocess(wrapped_code, timeout, work_dir)

    with DockerSandbox(work_dir=work_dir) as sandbox:
        wrapped_code = EXECUTION_TEMPLATE.format(
            train_path="/workspace/input/train.csv",
            test_path="/workspace/input/test.csv",
            target_cols=["formation_energy_ev_natom", "bandgap_energy_ev"],
            user_code=code,
        )
        return sandbox.run(wrapped_code, timeout=timeout)


def _execute_subprocess(
    wrapped_code: str, timeout: int, work_dir: str | None = None
) -> Dict:
    """Execute wrapped code in subprocess (internal helper).

    Returns Dict with keys: stdout, stderr, exit_code, score, status, traceback.
    Uses Popen to capture partial stdout/stderr on timeout.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="mle_eval_"
    ) as f:
        f.write(wrapped_code)
        script_path = f.name

    cwd = work_dir or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    stdout = ""
    stderr = ""
    exit_code = -1
    status = "error"

    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env={
                **os.environ,
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "PYTHONUNBUFFERED": "1",
            },
            preexec_fn=_set_resource_limits,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            exit_code = proc.returncode if proc.returncode is not None else -1
            status = "ok" if exit_code == 0 else "error"
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                partial_stdout, partial_stderr = proc.communicate(timeout=5)
                stdout = partial_stdout if partial_stdout else ""
                stderr = partial_stderr if partial_stderr else ""
            except Exception:
                stdout = ""
                stderr = ""
            exit_code = -1
            status = "timeout"
            stderr = f"Execution timed out after {timeout} seconds\n{stderr}"
            log_node_event(
                "_execute_subprocess",
                "timeout",
                {
                    "timeout": timeout,
                    "partial_stdout_len": len(stdout),
                    "partial_stderr_len": len(stderr),
                    "partial_stdout_preview": stdout[:500],
                },
            )
    except Exception as e:
        log_node_event("_execute_subprocess", "error", {"error": str(e)[:200]})
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "score": None,
            "status": "error",
            "traceback": "",
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    score = parse_score(stdout)

    log_node_event(
        "_execute_subprocess",
        "result",
        {
            "exit_code": exit_code,
            "score": score,
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "stdout_preview": stdout[:500] if stdout else "",
            "stderr_preview": stderr[:500] if stderr else "",
            "status": "ok" if exit_code == 0 and score is not None else status,
        },
    )

    if exit_code != 0 or score is None:
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "score": None,
            "status": status if status == "timeout" else "error",
            "traceback": _extract_traceback(stderr),
        }

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "score": score,
        "status": "ok",
        "traceback": "",
    }


FORBIDDEN_MODULES = {
    "os.system",
    "os.popen",
    "subprocess",
    "shutil.rmtree",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "requests",
    "pathlib.Path.unlink",
}

FORBIDDEN_CALLS = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "type.__subclasses__",
}

ALLOWED_OPEN_PATTERNS = [
    r"open\(['\"]train\.csv",
    r"open\(['\"]test\.csv",
    r"open\(['\"]sample_submission\.csv",
    r"\.read_csv\(",
    r"\.to_csv\(",
]


def _strip_markdown_fences(code: str) -> str:
    """Strip markdown code fences from LLM output before AST parsing or execution.

    Handles:
      - ```python ... ```  (standard)
      - ``` ... ```         (no language tag)
      - Unclosed fences: ```python ...  (LLM truncation)
      - Bare code           (no fences at all)
    """
    if not code or not code.strip():
        return code
    text = code.strip()
    m = re.search(r"```(?:python|py)\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python|py)?\s*(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        if content and not content.startswith("```"):
            return content
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:]
        if inner.endswith("```"):
            inner = inner[:-3]
        inner = inner.strip()
        for lang_prefix in ("python", "py"):
            if inner.startswith(lang_prefix):
                inner = inner[len(lang_prefix) :].strip()
                break
        return inner
    # Handle unclosed fences: ```python\n... (no closing ```)
    if text.startswith("```"):
        inner = text[3:]
        stripped_lang = False
        for lang_prefix in ("python", "py"):
            if inner.startswith(lang_prefix):
                inner = inner[len(lang_prefix) :].strip()
                stripped_lang = True
                break
        if not stripped_lang and (inner.startswith("\n") or inner.startswith("\r")):
            inner = inner.strip()
        if inner and not inner.startswith("```"):
            return inner
    return code


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """AST-based safety check for candidate code before execution.

    Blocks dangerous calls and imports while allowing sklearn, pandas, numpy, etc.
    Allows read-only file access for train.csv/test.csv.

    Args:
        code: Python code string to validate.

    Returns:
        Tuple of (is_safe, reason). is_safe=True if code passes all checks.
    """
    if not code or not code.strip():
        return False, "Empty code"

    code = _strip_markdown_fences(code)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in ("subprocess", "ctypes", "socket", "http", "urllib"):
                    return False, f"Forbidden import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module.split(".")[0] if node.module else ""
            if module_name in ("subprocess", "ctypes", "socket", "http", "urllib"):
                return False, f"Forbidden import from: {node.module}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ("eval", "exec", "compile", "__import__"):
                    return False, f"Forbidden function call: {func_name}"

            elif isinstance(node.func, ast.Attribute):
                attr_chain = _get_attr_chain(node.func)
                if attr_chain:
                    if attr_chain in FORBIDDEN_MODULES:
                        return False, f"Forbidden method call: {attr_chain}"

    lines = code.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "os.system(" in stripped or "os.popen(" in stripped:
            return False, f"Line {i + 1}: forbidden os call"

    return True, "safe"


def _get_attr_chain(node: ast.Attribute) -> Optional[str]:
    """Extract attribute chain from ast.Attribute node (e.g., 'os.system')."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


EXECUTION_TEMPLATE = """
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Data Loading ──
train = pd.read_csv({train_path!r})
test = pd.read_csv({test_path!r})

# ── Target and Feature Setup ──
target_cols = {target_cols}
feature_cols = [c for c in train.columns if c not in target_cols and c != 'id']
X = train[feature_cols].values
Y = train[target_cols].values
X_test = test[feature_cols].values

# ── User Code ──
{user_code}

# ── Evaluation (fallback) ──
# If the user code defines predict_model or train_and_predict but did NOT
# print a score, run cross-validation evaluation using those functions.
# Self-contained code that prints its own score does not need this block.
_has_api_functions = 'predict_model' in dir() or 'train_and_predict' in dir()

if _has_api_functions:
    from sklearn.model_selection import KFold

    def rmsle(y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 0, None)
        y_true_clipped = np.clip(y_true, 0, None)
        return np.sqrt(np.mean((np.log1p(y_pred_clipped) - np.log1p(y_true_clipped)) ** 2))

    def evaluate_model():
        _eval_train = pd.read_csv({train_path!r})
        _tcols = {target_cols}
        _fcols = [c for c in _eval_train.columns if c not in _tcols and c != 'id']
        _X = _eval_train[_fcols].values
        _Y = _eval_train[_tcols].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_scores = []
        for target_idx in range(_Y.shape[1]):
            y_col = _Y[:, target_idx]
            col_scores = []
            for train_idx, val_idx in kf.split(_X):
                X_tr, X_val = _X[train_idx], _X[val_idx]
                y_tr, y_val = y_col[train_idx], y_col[val_idx]
                try:
                    if 'predict_model' in dir():
                        y_pred = predict_model(X_tr, y_tr, X_val)
                    elif 'train_and_predict' in dir():
                        y_pred = train_and_predict(X_tr, y_tr, X_val)
                    else:
                        y_pred = np.zeros(len(X_val))
                    col_score = rmsle(y_val, y_pred)
                except Exception as e:
                    print(f"VALIDATION_ERROR: {{e}}", file=sys.stderr)
                    col_score = 9.999
                col_scores.append(col_score)
            all_scores.append(np.mean(col_scores))
        return np.mean(all_scores)

    try:
        final_score = evaluate_model()
        print(f"Final Validation Performance: {{final_score:.6f}}")
    except Exception as e:
        print(f"EXECUTION_ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)
"""


def execute_code(
    code: str,
    train_path: str = "input/train.csv",
    test_path: str = "input/test.csv",
    target_cols: Optional[List[str]] = None,
    timeout: Optional[int] = None,
    work_dir: Optional[str] = None,
    use_docker: Optional[bool] = None,
) -> Dict:
    """Execute Python code and parse the results.

    Stage 10: Supports Docker sandbox execution. When use_docker is True
    and Docker is available, code runs in an ephemeral container. Otherwise
    falls back to subprocess execution.

    Validates code safety (AST-based), wraps in execution template, runs,
    and parses the output for the score.

    Args:
        code: Python code string to execute.
        train_path: Path to training CSV.
        test_path: Path to test CSV.
        target_cols: List of target column names.
        timeout: Timeout in seconds (default: from config).
        work_dir: Working directory for execution.
        use_docker: If True, use Docker sandbox. If None (default), auto-detect
            from DOCKER_SANDBOX_ENABLED config. If False, always use subprocess.

    Returns:
        Dict with keys: stdout, stderr, exit_code, score, status, traceback.
        `traceback` contains the formatted traceback from stderr when execution
        fails, or an empty string on success.
    """
    if target_cols is None:
        target_cols = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    exec_timeout = timeout or EXECUTION_TIMEOUT

    code = _strip_markdown_fences(code)

    should_use_docker = use_docker if use_docker is not None else DOCKER_SANDBOX_ENABLED

    is_safe, reason = validate_code_safety(code)
    if not is_safe:
        log_node_event(
            "execute_code", "safety_check", {"status": "blocked", "reason": reason}
        )
        return {
            "stdout": "",
            "stderr": f"Code safety check failed: {reason}",
            "exit_code": -1,
            "score": None,
            "status": "blocked",
            "traceback": "",
        }

    if should_use_docker:
        docker_train = (
            "/workspace/input/train.csv"
            if train_path == "input/train.csv"
            else train_path
        )
        docker_test = (
            "/workspace/input/test.csv" if test_path == "input/test.csv" else test_path
        )

        wrapped_code = EXECUTION_TEMPLATE.format(
            train_path=docker_train,
            test_path=docker_test,
            target_cols=target_cols,
            user_code=code,
        )

        if _docker_available():
            with DockerSandbox(work_dir=work_dir or os.getcwd()) as sandbox:
                return sandbox.run(wrapped_code, timeout=exec_timeout)

        log_node_event(
            "execute_code",
            "docker_fallback",
            {"reason": "Docker unavailable, using subprocess"},
        )

    wrapped_code = EXECUTION_TEMPLATE.format(
        train_path=train_path,
        test_path=test_path,
        target_cols=target_cols,
        user_code=code,
    )

    result = _execute_subprocess(wrapped_code, exec_timeout, work_dir)

    _log_execution_artifact(code, wrapped_code, result, exec_timeout, work_dir)

    return result


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


@traceable("eval_ensemble")
def eval_ensemble(state: dict) -> dict:
    """Execute the ensemble code and evaluate its performance.

    Substitutes the ensemble code and runs it. If execution fails and
    debug retries are available, routes to debug.

    Mock: generates a score slightly better than best_ensemble_score,
    10% chance of execution error.
    """
    if _is_mock_mode():
        return _eval_ensemble_mock(state)
    return _eval_ensemble_real(state)


def _eval_ensemble_mock(state: dict) -> dict:
    simulate_delay()

    best_ensemble_score = state.get("best_ensemble_score", 0)
    ensemble_round = state.get("ensemble_round", 0)
    debug_retries = state.get("debug_retries", 0)

    crashed = random_pass(0.10)

    if crashed and debug_retries < MAX_DEBUG_RETRIES:
        log_node_event(
            "eval_ensemble",
            "output",
            {"status": "error", "debug_retries": debug_retries},
        )
        return {
            "execution_output": "",
            "execution_error": "Mock ensemble execution error",
            "execution_exit_code": -1,
            "execution_score": None,
            "status": "error",
        }

    improvement = random_score(0.001, 0.02)
    metric_direction = state.get("metric_direction", "maximize")
    score = (
        best_ensemble_score + improvement + ensemble_round * 0.005
        if metric_direction == "maximize"
        else best_ensemble_score - improvement - ensemble_round * 0.005
    )

    current_code = state.get("current_ensemble_code", "")

    improved_score = (
        score
        if normalize_score(score, metric_direction)
        > normalize_score(best_ensemble_score, metric_direction)
        else best_ensemble_score
    )
    improved_code = (
        current_code
        if normalize_score(score, metric_direction)
        >= normalize_score(best_ensemble_score, metric_direction)
        else state.get("best_ensemble_code", "")
    )

    log_node_event(
        "eval_ensemble",
        "output",
        {
            "status": "ok",
            "score": round(score, 4),
            "best_ensemble_score": round(improved_score, 4),
            "ensemble_round": ensemble_round,
        },
    )

    return {
        "execution_output": f"Final Validation Performance: {score:.4f}",
        "execution_error": None,
        "execution_score": round(score, 4),
        "current_ensemble_score": round(score, 4),
        "best_ensemble_code": improved_code,
        "best_ensemble_score": round(improved_score, 4),
        "status": "ok",
    }


def _eval_ensemble_real(state: dict) -> dict:
    best_ensemble_score = state.get("best_ensemble_score", 0)
    best_ensemble_code = state.get("best_ensemble_code", "")
    current_ensemble_code = state.get("current_ensemble_code", "")
    debug_retries = state.get("debug_retries", 0)
    metric_direction = state.get("metric_direction", "maximize")

    result = execute_code(
        current_ensemble_code,
        train_path=state.get("train_path", "input/train.csv"),
        test_path=state.get("test_path", "input/test.csv"),
        target_cols=state.get("target_cols"),
    )

    if result["status"] != "ok" or result["score"] is None:
        if debug_retries < MAX_ENSEMBLE_DEBUG_RETRIES:
            log_node_event(
                "eval_ensemble",
                "output",
                {"status": "error", "debug_retries": debug_retries},
            )
            return {
                "execution_output": result.get("stdout", ""),
                "execution_error": result.get("stderr", "Execution failed"),
                "execution_exit_code": result.get("exit_code", -1),
                "execution_score": None,
                "status": "error",
            }
        log_node_event(
            "eval_ensemble",
            "output",
            {"status": "error", "debug_retries_exhausted": True},
        )
        return {
            "execution_output": result.get("stdout", ""),
            "execution_error": result.get("stderr", "Execution failed"),
            "execution_exit_code": result.get("exit_code", -1),
            "execution_score": None,
            "status": "error",
        }

    score = result["score"]

    if normalize_score(score, metric_direction) > normalize_score(
        best_ensemble_score, metric_direction
    ):
        new_best_score = score
        new_best_code = current_ensemble_code
    else:
        new_best_score = best_ensemble_score
        new_best_code = best_ensemble_code

    log_node_event(
        "eval_ensemble",
        "output",
        {
            "status": "ok",
            "score": round(score, 4),
            "best_ensemble_score": round(new_best_score, 4),
        },
    )

    return {
        "execution_output": result.get("stdout", ""),
        "execution_error": None,
        "execution_score": round(score, 4),
        "current_ensemble_score": round(score, 4),
        "best_ensemble_code": new_best_code,
        "best_ensemble_score": round(new_best_score, 4),
        "status": "ok",
    }


@traceable("eval_submission")
def eval_submission(state: dict) -> dict:
    """Execute the final submission script and evaluate its score.

    Runs the submission code (with subsampling removed) on the full dataset.

    Mock: returns a score slightly better than best_score.
    """
    if _is_mock_mode():
        return _eval_submission_mock(state)
    return _eval_submission_real(state)


def _write_final_artifacts(
    run_dir: str, submission_code: str, submission_score: float | None
) -> str | None:
    """Write submission artifacts to {run_dir}/final/ directory.

    Creates the final/ directory and writes:
    - submission_code.py: the final submission script
    - submission_score.txt: the validation score

    Returns the final directory path, or None if run_dir is unavailable.
    """
    if not run_dir:
        return None

    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    code_path = os.path.join(final_dir, "submission_code.py")
    with open(code_path, "w") as f:
        f.write(submission_code)

    score_path = os.path.join(final_dir, "submission_score.txt")
    with open(score_path, "w") as f:
        f.write(f"{submission_score}\n" if submission_score is not None else "None\n")

    log_node_event(
        "eval_submission",
        "final_artifacts",
        {"final_dir": final_dir, "code_path": code_path, "score_path": score_path},
    )

    return final_dir


def _eval_submission_mock(state: dict) -> dict:
    simulate_delay()

    best_score = state.get("best_score", 0)
    metric_direction = state.get("metric_direction", "maximize")

    improvement = random_score(0.001, 0.005)
    score = (
        best_score + improvement
        if metric_direction == "maximize"
        else best_score - improvement
    )

    submission_code = state.get("submission_code", "")

    run_dir = get_run_dir()
    final_dir = _write_final_artifacts(run_dir, submission_code, round(score, 4))

    log_node_event(
        "eval_submission",
        "output",
        {"status": "ok", "score": round(score, 4), "final_dir": final_dir},
    )

    return {
        "submission_score": round(score, 4),
        "status": "ok",
        "final_dir": final_dir or "",
    }


def _eval_submission_real(state: dict) -> dict:
    submission_code = state.get("submission_code", "")

    result = execute_code(
        submission_code,
        train_path=state.get("train_path", "input/train.csv"),
        test_path=state.get("test_path", "input/test.csv"),
        target_cols=state.get("target_cols"),
    )

    if result["status"] != "ok" or result["score"] is None:
        log_node_event(
            "eval_submission",
            "output",
            {"status": "error"},
        )
        return {
            "submission_score": None,
            "status": "error",
            "final_dir": "",
        }

    score = result["score"]

    run_dir = get_run_dir()
    final_dir = _write_final_artifacts(run_dir, submission_code, round(score, 4))

    log_node_event(
        "eval_submission",
        "output",
        {"status": "ok", "score": round(score, 4), "final_dir": final_dir},
    )

    return {
        "submission_score": round(score, 4),
        "status": "ok",
        "final_dir": final_dir or "",
    }
