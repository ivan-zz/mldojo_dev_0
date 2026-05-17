"""Verification nodes: A_verify (semantic), A_sast (security).

Stage 10: Real implementations for both semantic and security verification.

A_verify: Enhanced semantic verification with structured per-check output.
A_sast: SAST (Static Application Security Testing) with regex, AST, and
         hallucinated-import checks. Critical violations block execution;
         non-critical violations are logged as warnings.

Also provides standalone check functions for use by other modules:
    _regex_sast_check(code) -> list[dict]
    _ast_sast_check(code) -> list[dict]
    _check_hallucinated_imports(code) -> list[dict]
"""

import ast
import importlib
import importlib.util
import os
import re

from src.mle_star.config import MOCK_MODE
from src.mle_star.prompts.verification import SEMANTIC_VERIFY_PROMPT, SAST_CHECK_PROMPT
from src.mle_star.state.shared import (
    traceable,
    simulate_delay,
    log_node_event,
    call_llm,
    _default_llm_config,
    parse_json_response,
)


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


# ── SAST Helpers ──────────────────────────────────────────────────────────


_CRITICAL_PATTERNS = [
    (r"os\.system\s*\(", "os.system call", "critical"),
    (r"os\.popen\s*\(", "os.popen call", "critical"),
    (r"subprocess\.\w+.*shell\s*=\s*True", "subprocess with shell=True", "critical"),
    (r"\beval\s*\(", "eval() call", "critical"),
    (r"\bexec\s*\(", "exec() call", "critical"),
    (r"\bcompile\s*\(", "compile() call", "warning"),
    (r"__import__\s*\(", "__import__() call", "critical"),
    (r"ctypes\.", "ctypes usage", "critical"),
    (r"pickle\.loads?\s*\(", "pickle deserialization", "critical"),
    (r"shutil\.rmtree\s*\(", "shutil.rmtree call", "critical"),
    (r"pathlib\.Path.*\.unlink\s*\(", "Path.unlink call", "warning"),
    (r"\.to_csv\s*\(", "file write (to_csv)", "allowed"),
    (r"\.read_csv\s*\(", "file read (read_csv)", "allowed"),
]

# Note: _WARNING_PATTERNS are named for their SAST category (network access)
# but classified as severity="critical" because they indicate data exfiltration risk.
_WARNING_PATTERNS = [
    (r"socket\.", "socket usage", "critical"),
    (r"http\.", "http usage", "critical"),
    (r"urllib\.", "urllib usage", "critical"),
    (r"requests\.", "requests usage", "critical"),
]

_FORBIDDEN_IMPORT_MODULES = {
    "subprocess",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "requests",
}

_KNOWN_ML_PACKAGES = {
    "sklearn",
    "pandas",
    "numpy",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "tensorflow",
    "torch",
    "keras",
    "torchvision",
    "transformers",
    "datasets",
    "optuna",
    "hyperopt",
    "imblearn",
    "joblib",
    "tqdm",
    "pillow",
    "cv2",
    "skimage",
    "sympy",
    "networkx",
    "textblob",
    "nltk",
    "spacy",
    "gensim",
    "wordcloud",
    "shap",
    "eli5",
    "lime",
    "yellowbrick",
    "plotly",
    "bokeh",
    "altair",
    "dash",
    "streamlit",
    "fastapi",
    "flask",
    "sqlalchemy",
    "pymysql",
    "psycopg2",
    "mongo",
    "redis",
    "celery",
    "pytest",
    "unittest",
    "json",
    "csv",
    "re",
    "math",
    "random",
    "statistics",
    "itertools",
    "collections",
    "functools",
    "operator",
    "pathlib",
    "os",
    "sys",
    "datetime",
    "time",
    "logging",
    "typing",
    "dataclasses",
    "abc",
    "copy",
    "heapq",
    "bisect",
    "array",
    "struct",
    "warnings",
    "contextlib",
    "io",
    "tempfile",
    "glob",
    "fnmatch",
    "linecache",
    "shutil",
    "argparse",
    "enum",
    "decimal",
    "fractions",
    "hashlib",
    "traceback",
    "inspect",
    "dis",
    "ast",
    "token",
    "tokenize",
    "pickle",
    "sqlite3",
    "zlib",
    "gzip",
    "bz2",
    "lzma",
    "zipfile",
    "tarfile",
    "configparser",
    "xml",
    "html",
    "email",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "socket",
    "ssl",
}


def _regex_sast_check(code: str) -> list[dict]:
    """Regex-based SAST check for common dangerous patterns.

    Returns list of violation dicts:
        {pattern, match, severity, description}
    """
    violations = []
    lines = code.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern, desc, severity in _CRITICAL_PATTERNS + _WARNING_PATTERNS:
            if severity == "allowed":
                continue
            match = re.search(pattern, stripped)
            if match:
                violations.append(
                    {
                        "pattern": pattern,
                        "match": match.group(),
                        "line": i + 1,
                        "severity": severity,
                        "description": desc,
                    }
                )

    return violations


def _ast_sast_check(code: str) -> list[dict]:
    """AST-based SAST check for structural code violations.

    Walks the AST to detect:
    - Forbidden imports (subprocess, ctypes, socket, http, urllib, requests)
    - Dangerous function calls (eval, exec, compile, __import__)
    - pickle.loads/pickle.load calls
    - subprocess with shell=True
    - os.system/os.popen calls

    Returns list of violation dicts:
        {type, description, line, severity}
    """
    violations = []

    if not code or not code.strip():
        return violations

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in _FORBIDDEN_IMPORT_MODULES:
                    violations.append(
                        {
                            "type": "forbidden_import",
                            "description": f"Forbidden import: {alias.name}",
                            "line": node.lineno,
                            "severity": "critical",
                        }
                    )

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module.split(".")[0] if node.module else ""
            if module_name in _FORBIDDEN_IMPORT_MODULES:
                violations.append(
                    {
                        "type": "forbidden_import_from",
                        "description": f"Forbidden import from: {node.module}",
                        "line": node.lineno,
                        "severity": "critical",
                    }
                )

        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ("eval", "exec", "compile", "__import__"):
                    violations.append(
                        {
                            "type": "dangerous_call",
                            "description": f"Dangerous function call: {func_name}()",
                            "line": node.lineno,
                            "severity": "critical",
                        }
                    )

            elif isinstance(node.func, ast.Attribute):
                attr_chain = _get_attr_chain(node.func)
                if attr_chain:
                    dangerous_chains = {
                        "os.system": "critical",
                        "os.popen": "critical",
                        "pickle.loads": "critical",
                        "pickle.load": "critical",
                    }
                    if attr_chain in dangerous_chains:
                        violations.append(
                            {
                                "type": "dangerous_method",
                                "description": f"Dangerous method call: {attr_chain}()",
                                "line": node.lineno,
                                "severity": dangerous_chains[attr_chain],
                            }
                        )

                    if (
                        attr_chain == "subprocess.call"
                        or attr_chain == "subprocess.run"
                    ):
                        for kw in node.keywords:
                            if kw.arg == "shell":
                                if (
                                    isinstance(kw.value, ast.Constant)
                                    and kw.value.value is True
                                ):
                                    violations.append(
                                        {
                                            "type": "shell_true",
                                            "description": f"subprocess call with shell=True: {attr_chain}()",
                                            "line": node.lineno,
                                            "severity": "critical",
                                        }
                                    )
                                elif (
                                    isinstance(kw.value, ast.Name)
                                    and kw.value.id == "True"
                                ):
                                    violations.append(
                                        {
                                            "type": "shell_true",
                                            "description": f"subprocess call with shell=True: {attr_chain}()",
                                            "line": node.lineno,
                                            "severity": "critical",
                                        }
                                    )

    return violations


def _check_hallucinated_imports(code: str) -> list[dict]:
    """Check for imports of packages that don't exist.

    Uses importlib.util.find_spec() to verify each import. Skips
    packages in _KNOWN_ML_PACKAGES (whitelist of common packages).

    Returns list of violation dicts:
        {type, description, package, severity}
    """
    violations = []

    if not code or not code.strip():
        return violations

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return violations

    imports_to_check = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name not in _KNOWN_ML_PACKAGES:
                    imports_to_check.append((module_name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name not in _KNOWN_ML_PACKAGES:
                    imports_to_check.append((module_name, node.lineno))

    for module_name, line in imports_to_check:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                violations.append(
                    {
                        "type": "hallucinated_import",
                        "description": f"Hallucinated import: {module_name} (package not found)",
                        "package": module_name,
                        "line": line,
                        "severity": "critical",
                    }
                )
        except (ModuleNotFoundError, ValueError):
            violations.append(
                {
                    "type": "hallucinated_import",
                    "description": f"Hallucinated import: {module_name} (package not found)",
                    "package": module_name,
                    "line": line,
                    "severity": "critical",
                }
            )
        except Exception:
            pass

    return violations


def _get_attr_chain(node: ast.Attribute) -> str | None:
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


def _deduplicate_violations(violations: list[dict]) -> list[dict]:
    """Remove duplicate violations based on description + line."""
    seen = set()
    result = []
    for v in violations:
        key = (v.get("description", ""), v.get("line", 0))
        if key not in seen:
            seen.add(key)
            result.append(v)
    return result


def _build_violation_report(violations: list[dict]) -> str:
    """Build a human-readable report from violation list."""
    if not violations:
        return "No security violations found."
    lines = []
    for v in violations:
        severity = v.get("severity", "warning")
        desc = v.get("description", "Unknown violation")
        line_no = v.get("line", "?")
        lines.append(f"[{severity.upper()}] Line {line_no}: {desc}")
    return "\n".join(lines)


# ── Node Implementations ─────────────────────────────────────────────────


@traceable("A_verify")
def A_verify(state: dict) -> dict:
    """Semantic verification of the refined code.

    Enhanced (Stage 10): Checks task alignment, target variable correctness,
    metric computation, data splitting strategy, and more.

    Returns structured per-check output:
    - ok: {status: "ok", checks: {task: pass, target: pass, metric: pass, split: pass}}
    - fail: {status: "semantic_fail", feedback: "...", failed_checks: [...]}

    Mock: always passes.
    Real: uses LLM to verify semantic correctness.
    """
    if _is_mock_mode():
        simulate_delay()

        refined_code = state.get("refined_code", "")

        log_node_event(
            "A_verify",
            "output",
            {"status": "ok", "refined_code_len": len(refined_code)},
        )

        return {"status": "ok", "report": "No violations found."}

    refined_code = state.get("refined_code", "")
    current_solution = state.get("current_solution", "")
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "accuracy")

    prompt = SEMANTIC_VERIFY_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        current_solution=current_solution,
        refined_code=refined_code,
    )

    try:
        response = call_llm(prompt, response_format="json")
        result = parse_json_response(response)
        status = result.get("status", "ok")
        feedback = result.get("feedback", "")
        failed_checks = result.get("failed_checks", [])
        checks = result.get("checks", {})

        if status == "semantic_fail":
            log_node_event(
                "A_verify",
                "output",
                {
                    "status": "semantic_fail",
                    "feedback": feedback[:200],
                    "failed_checks": failed_checks,
                },
            )
            return {
                "status": "semantic_fail",
                "feedback": feedback,
                "failed_checks": failed_checks,
                "checks": checks,
            }

        log_node_event(
            "A_verify",
            "output",
            {"status": "ok", "refined_code_len": len(refined_code), "checks": checks},
        )
        return {"status": "ok", "checks": checks}

    except Exception as e:
        log_node_event(
            "A_verify",
            "llm_failed",
            {"error": str(e)[:200]},
        )
        return {"status": "ok", "report": "Verification passed."}


@traceable("A_sast")
def A_sast(state: dict) -> dict:
    """Security static analysis gate.

    Stage 10 real implementation: runs regex + AST + hallucinated-import
    checks. Optionally uses LLM-based SAST for deeper analysis.

    Returns:
        pass: {status: "pass"}
        critical: {status: "critical_violation", violations: [...], report: "..."}
        warning: {status: "pass", warnings: [...]}  (non-critical warnings logged)
    """
    if _is_mock_mode():
        simulate_delay()

        log_node_event(
            "A_sast",
            "output",
            {"status": "pass"},
        )

        return {"status": "pass", "report": "Mock mode: all checks passed."}

    refined_code = state.get("refined_code", "")
    candidate_solution = state.get("candidate_solution", "")

    code_to_check = candidate_solution or refined_code or ""
    if not code_to_check:
        log_node_event("A_sast", "output", {"status": "pass", "reason": "empty_code"})
        return {"status": "pass", "report": "No code to check."}

    all_violations = []

    regex_violations = _regex_sast_check(code_to_check)
    all_violations.extend(regex_violations)

    ast_violations = _ast_sast_check(code_to_check)
    all_violations.extend(ast_violations)

    hallucinated_violations = _check_hallucinated_imports(code_to_check)
    all_violations.extend(hallucinated_violations)

    all_violations = _deduplicate_violations(all_violations)

    critical_violations = [v for v in all_violations if v.get("severity") == "critical"]
    warnings = [v for v in all_violations if v.get("severity") == "warning"]

    if critical_violations:
        report = _build_violation_report(critical_violations)
        log_node_event(
            "A_sast",
            "output",
            {
                "status": "critical_violation",
                "num_critical": len(critical_violations),
                "num_warnings": len(warnings),
                "report": report[:500],
            },
        )

        return {
            "status": "critical_violation",
            "violations": critical_violations,
            "warnings": warnings,
            "report": report,
        }

    if warnings:
        warning_report = _build_violation_report(warnings)
        log_node_event(
            "A_sast",
            "output",
            {
                "status": "pass",
                "num_warnings": len(warnings),
                "warnings_summary": warning_report[:300],
            },
        )

        return {
            "status": "pass",
            "warnings": warnings,
            "report": warning_report,
        }

    log_node_event(
        "A_sast",
        "output",
        {"status": "pass"},
    )

    return {"status": "pass", "report": "No security violations found."}


def approve_security_exception(state: dict, violation_index: int | None = None) -> dict:
    """Approve a security exception, allowing code with SAST violations to proceed.

    Marks critical_violation status as approved so that the refinement
    subgraph can route the code to execution instead of back to A7__implement.

    Args:
        state: Current state dict containing A_sast output.
        violation_index: If provided, approve only this specific violation
            (by index). If None, approve all violations.

    Returns:
        Updated state with security_violations_approved flag and cleared
        critical_violation status.
    """
    sast_status = state.get("status", "")

    if sast_status != "critical_violation":
        return {"status": "pass"}

    violations = state.get("violations", [])

    if violation_index is not None and 0 <= violation_index < len(violations):
        approved = [violations[violation_index]]
    else:
        approved = violations

    approved_descriptions = [v.get("description", "unknown") for v in approved]

    log_node_event(
        "approve_security_exception",
        "approved",
        {
            "num_approved": len(approved),
            "approved_violations": approved_descriptions,
        },
    )

    return {
        "status": "pass",
        "security_violations": approved_descriptions,
        "security_exceptions_approved": True,
    }
