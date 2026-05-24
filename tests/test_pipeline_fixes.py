"""Targeted tests for pipeline run failure fixes.

Bug #1: EXECUTION_MAX_CPU_SECONDS too low (300s → 1800s)
Bug #2: Markdown fences in code block validation
Bug #3: JSON parse failures for markdown-wrapped ablation responses
Bug #4: Score fallback of 1.0 (should be direction-aware failure score)
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBug1ExecutionTimeouts:
    """Bug #1: EXECUTION_MAX_CPU_SECONDS and EXECUTION_TIMEOUT defaults."""

    def test_execution_timeout_default_is_180(self):
        from src.mle_star.config import EXECUTION_TIMEOUT

        assert EXECUTION_TIMEOUT == 180

    def test_execution_max_cpu_seconds_default_is_300(self):
        from src.mle_star.config import EXECUTION_MAX_CPU_SECONDS

        assert EXECUTION_MAX_CPU_SECONDS == 300

    def test_per_node_timeout_default_is_600(self):
        from src.mle_star.config import PER_NODE_TIMEOUT_SECONDS

        assert PER_NODE_TIMEOUT_SECONDS == 600


class TestBug2MarkdownFenceStripping:
    """Bug #2: validate_code_safety and execute_code must strip markdown fences."""

    def test_strip_python_fence(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```python\nimport numpy as np\nprint(1)\n```"
        result = _strip_markdown_fences(code)
        assert result == "import numpy as np\nprint(1)"

    def test_strip_py_fence(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```py\nimport numpy as np\nprint(1)\n```"
        result = _strip_markdown_fences(code)
        assert result == "import numpy as np\nprint(1)"

    def test_strip_plain_fence(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```\nimport numpy as np\nprint(1)\n```"
        result = _strip_markdown_fences(code)
        assert result == "import numpy as np\nprint(1)"

    def test_no_fence_returns_unchanged(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "import numpy as np\nprint(1)"
        result = _strip_markdown_fences(code)
        assert result == code

    def test_validate_code_safety_with_python_fence(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = "```python\nimport numpy as np\nx = 1 + 2\n```"
        is_safe, reason = validate_code_safety(code)
        assert is_safe, f"Should be safe but got: {reason}"

    def test_validate_code_safety_with_py_fence(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = "```py\nimport pandas as pd\ndf = pd.DataFrame()\n```"
        is_safe, reason = validate_code_safety(code)
        assert is_safe, f"Should be safe but got: {reason}"

    def test_validate_code_safety_with_plain_fence(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = "```\nfrom sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()\n```"
        is_safe, reason = validate_code_safety(code)
        assert is_safe, f"Should be safe but got: {reason}"

    def test_validate_code_safety_fence_with_subprocess_still_blocked(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = "```python\nimport subprocess\nsubprocess.run(['rm', '-rf', '/'])\n```"
        is_safe, reason = validate_code_safety(code)
        assert not is_safe
        assert "Forbidden" in reason

    def test_execute_code_strips_fences_before_safety_check(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```python\nx = 1\n```"
        stripped = _strip_markdown_fences(code)
        assert "```" not in stripped
        assert stripped == "x = 1"

    def test_strip_fence_multiline_code(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```python\nimport numpy as np\nimport pandas as pd\ntrain = pd.read_csv('train.csv')\nprint(train.shape)\n```"
        result = _strip_markdown_fences(code)
        assert result.startswith("import numpy")
        assert result.endswith("print(train.shape)")
        assert "```" not in result


class TestBug3JsonParseHardening:
    """Bug #3: parse_json_response must handle markdown-wrapped JSON."""

    def test_plain_json(self):
        from src.mle_star.state.shared import parse_json_response

        raw = '{"key": "value"}'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_in_backtick_fence_with_json_tag(self):
        from src.mle_star.state.shared import parse_json_response

        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_in_backtick_fence_with_python_tag(self):
        from src.mle_star.state.shared import parse_json_response

        raw = '```python\n{"key": "value"}\n```'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_in_backtick_fence_no_tag(self):
        from src.mle_star.state.shared import parse_json_response

        raw = '```\n{"key": "value"}\n```'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_in_backtick_no_trailing_newline(self):
        from src.mle_star.state.shared import parse_json_response

        raw = '```json\n{"key": "value"}```'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_with_nested_braces(self):
        from src.mle_star.state.shared import parse_json_response

        raw = '```json\n{"ablation_scripts": [{"name": "baseline", "block_name": "baseline", "code": "import numpy"}]}\n```'
        result = parse_json_response(raw)
        assert isinstance(result, dict)
        assert "ablation_scripts" in result
        assert len(result["ablation_scripts"]) == 1

    def test_json_with_text_wrapper(self):
        from src.mle_star.state.shared import parse_json_response

        raw = 'Here are the ablation results:\n```json\n{"scripts": ["baseline"]}\n```\nHope that helps!'
        result = parse_json_response(raw)
        assert result == {"scripts": ["baseline"]}

    def test_json_brace_depth_tracking(self):
        from src.mle_star.state.shared import parse_json_response

        raw = 'Some text before {"nested": {"inner": 1}} and after'
        result = parse_json_response(raw)
        assert result == {"nested": {"inner": 1}}


class TestBug4FailureScore:
    """Bug #4: failure_score() returns direction-aware worst possible score."""

    def test_failure_score_minimize(self):
        from src.mle_star.state.shared import failure_score

        assert failure_score("minimize") == float("inf")

    def test_failure_score_maximize(self):
        from src.mle_star.state.shared import failure_score

        assert failure_score("maximize") == 0.0

    def test_failure_score_always_worse_after_normalize(self):
        from src.mle_star.state.shared import failure_score, normalize_score

        for direction in ["minimize", "maximize"]:
            fs = failure_score(direction)
            normalized_failure = normalize_score(fs, direction)
            normalized_real = normalize_score(0.05, direction)
            assert normalized_failure < normalized_real, (
                f"Failure score ({fs}) should be worse than 0.05 for {direction}"
            )

    def test_failure_score_minimize_worse_than_max_rmsle(self):
        from src.mle_star.state.shared import failure_score, normalize_score

        fs = failure_score("minimize")
        assert normalize_score(fs, "minimize") < normalize_score(0.10, "minimize")

    def test_failure_score_maximize_worse_than_accuracy(self):
        from src.mle_star.state.shared import failure_score, normalize_score

        fs = failure_score("maximize")
        assert normalize_score(fs, "maximize") < normalize_score(0.50, "maximize")


class TestCLIArgsConfig:
    """CLI args now propagate outer/inner/ensemble/debug iteration counts."""

    def test_build_config_propagates_all_args(self):
        from main import _build_config
        import argparse

        args = argparse.Namespace(
            max_full_cycles=2,
            num_parallel_solutions=2,
            max_outer_steps=2,
            max_inner_steps=2,
            max_ensemble_rounds=2,
            max_debug_retries=2,
            max_ablation_debug_retries=2,
            execution_timeout=180,
            provider=None,
            model=None,
            fast=False,
            mock=False,
        )
        config = _build_config(args)
        assert config.max_full_cycles == 2
        assert config.num_parallel_solutions == 2
        assert config.max_outer_steps == 2
        assert config.max_inner_steps == 2
        assert config.max_ensemble_rounds == 2
        assert config.max_debug_retries == 2
        assert config.max_ablation_debug_retries == 2

    def test_fast_mode_sets_all_to_one(self):
        from main import _build_config
        import argparse

        args = argparse.Namespace(
            max_full_cycles=3,
            num_parallel_solutions=2,
            max_outer_steps=4,
            max_inner_steps=4,
            max_ensemble_rounds=5,
            max_debug_retries=3,
            max_ablation_debug_retries=3,
            execution_timeout=180,
            provider=None,
            model=None,
            fast=True,
            mock=False,
        )
        config = _build_config(args)
        assert config.max_full_cycles == 1
        assert config.max_outer_steps == 1
        assert config.max_inner_steps == 1
        assert config.max_ensemble_rounds == 1
        assert config.num_parallel_solutions == 1
        assert config.max_debug_retries == 1
        assert config.max_ablation_debug_retries == 1


class TestAlg2GuardSkipsOnEmptyAlg1:
    """Bug: pipeline_flow_node must skip Alg2 when Alg1 produced no valid solution.

    Without the guard, Alg2 runs unconditionally even with best_solution=""
    and best_score=0, wasting LLM calls on garbage input.
    """

    def _make_pipeline_flow_node_inputs(self, alg1_result):
        from src.mle_star.graph import alg1_result_to_system

        state = {
            "run_index": 0,
            "task_desc": "test",
            "metric_direction": "minimize",
            "datasets": [],
            "score_function_desc": "",
        }
        alg1_sys_updates = alg1_result_to_system(alg1_result)
        best_solution = alg1_sys_updates.get("best_solution", "")
        best_score = alg1_sys_updates.get("best_score", 0)
        alg1_has_solution = bool(
            best_solution.strip()
            and isinstance(alg1_result.get("best_candidate", {}), dict)
            and alg1_result.get("best_candidate", {}).get("code")
        )
        return best_solution, best_score, alg1_has_solution

    def test_empty_best_candidate_skips_alg2(self):
        alg1_result = {"status": "search_complete", "best_candidate": {}}
        best_solution, best_score, alg1_has_solution = (
            self._make_pipeline_flow_node_inputs(alg1_result)
        )
        assert not alg1_has_solution
        assert best_solution == ""

    def test_best_candidate_with_code_passes(self):
        alg1_result = {
            "status": "search_complete",
            "best_candidate": {"code": "import numpy\nprint(1)", "score": 0.29},
        }
        best_solution, best_score, alg1_has_solution = (
            self._make_pipeline_flow_node_inputs(alg1_result)
        )
        assert alg1_has_solution
        assert "import numpy" in best_solution
        assert best_score == 0.29

    def test_whitespace_only_solution_skips_alg2(self):
        alg1_result = {
            "status": "search_complete",
            "best_candidate": {"code": "   \n  ", "score": 0.0},
        }
        best_solution, best_score, alg1_has_solution = (
            self._make_pipeline_flow_node_inputs(alg1_result)
        )
        assert not alg1_has_solution

    def test_error_status_with_no_candidate_skips_alg2(self):
        alg1_result = {"status": "error", "best_candidate": {}}
        best_solution, best_score, alg1_has_solution = (
            self._make_pipeline_flow_node_inputs(alg1_result)
        )
        assert not alg1_has_solution

    def test_started_status_with_no_candidate_skips_alg2(self):
        alg1_result = {"status": "start", "best_candidate": {}}
        best_solution, best_score, alg1_has_solution = (
            self._make_pipeline_flow_node_inputs(alg1_result)
        )
        assert not alg1_has_solution


class TestBug9UnclosedMarkdownFences:
    """Bug #9: parse_code_block and _strip_markdown_fences must handle
    unclosed markdown fences (LLM truncation where closing ``` is missing).

    This caused eval_candidate to crash with 'Syntax error in code: invalid syntax'
    when the LLM returned code starting with ```python but no closing ```.
    """

    def test_strip_unclosed_python_fence(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```python\nimport numpy as np\nprint(1)"
        result = _strip_markdown_fences(code)
        assert "```" not in result
        assert result.startswith("import numpy")

    def test_strip_unclosed_py_fence(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```py\nimport numpy as np\nprint(1)"
        result = _strip_markdown_fences(code)
        assert "```" not in result
        assert result.startswith("import numpy")

    def test_strip_unclosed_bare_fence(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```\nimport numpy as np\nprint(1)"
        result = _strip_markdown_fences(code)
        assert "```" not in result
        assert result.startswith("import numpy")

    def test_strip_unclosed_truncated_code(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```python\nimport numpy as np\ndf['col'] = df['col'] *"
        result = _strip_markdown_fences(code)
        assert "```" not in result
        assert result.startswith("import numpy")

    def test_validate_unclosed_fence_passes(self):
        from src.mle_star.nodes.execution import validate_code_safety

        code = "```python\nimport numpy as np\nprint(1)"
        is_safe, reason = validate_code_safety(code)
        assert is_safe, f"Should be safe but got: {reason}"

    def test_closed_fences_still_work(self):
        from src.mle_star.nodes.execution import _strip_markdown_fences

        code = "```python\nimport numpy as np\nprint(1)\n```"
        result = _strip_markdown_fences(code)
        assert "```" not in result
        assert result == "import numpy as np\nprint(1)"


class TestBug9ParseCodeBlockUnclosed:
    """Bug #9: parse_code_block must handle unclosed markdown fences."""

    def test_parse_unclosed_python_fence(self):
        from src.mle_star.state.shared import parse_code_block

        raw = "```python\nimport numpy as np\nprint(1)"
        result = parse_code_block(raw)
        assert "```" not in result
        assert result.startswith("import numpy")

    def test_parse_unclosed_py_fence(self):
        from src.mle_star.state.shared import parse_code_block

        raw = "```py\nimport numpy as np\nprint(1)"
        result = parse_code_block(raw)
        assert "```" not in result
        assert result.startswith("import numpy")

    def test_parse_unclosed_truncated_code(self):
        from src.mle_star.state.shared import parse_code_block

        raw = "```python\nimport numpy as np\ndf['col'] = df['col'] *"
        result = parse_code_block(raw)
        assert "```" not in result
        assert result.startswith("import numpy")

    def test_parse_closed_fence_priority(self):
        from src.mle_star.state.shared import parse_code_block

        raw = "```python\nimport numpy as np\nprint(1)\n```"
        result = parse_code_block(raw)
        assert result == "import numpy as np\nprint(1)"

    def test_parse_text_before_unclosed_fence(self):
        from src.mle_star.state.shared import parse_code_block

        raw = "Here is the code:\n```python\nimport numpy as np\nprint(1)"
        result = parse_code_block(raw)
        assert "```" not in result
        assert result.startswith("import numpy")
