"""Quick integration verification for all pipeline fixes."""

import os

os.environ["MLE_MOCK_MODE"] = "1"

from src.mle_star.state.alg1_state import CandidateState
from src.mle_star.subgraphs.candidate_subgraph import eval_candidate
from src.mle_star.nodes.execution import validate_code_safety, _strip_markdown_fences
from src.mle_star.state.shared import (
    failure_score,
    normalize_score,
    parse_code_block,
    parse_json_response,
)
from src.mle_star.config import EXECUTION_TIMEOUT, EXECUTION_MAX_CPU_SECONDS


def test_bug1_timeout_config():
    assert EXECUTION_TIMEOUT == 180
    assert EXECUTION_MAX_CPU_SECONDS == 300
    print("Bug #1 PASSED: EXECUTION_TIMEOUT=180, EXECUTION_MAX_CPU_SECONDS=300")


def test_bug2_fence_stripping():
    code = "import numpy as np\nx = np.array([1, 2, 3])\nprint(x.mean())"
    fenced_python = "```python\n" + code + "\n```"
    stripped = _strip_markdown_fences(fenced_python)
    assert stripped == code, f"Strip failed: got [{stripped[:50]}]"

    is_safe, reason = validate_code_safety(fenced_python)
    assert is_safe, f"Safety check failed on fenced code: {reason}"

    is_safe_bare, reason_bare = validate_code_safety(code)
    assert is_safe_bare, f"Safety check failed on bare code: {reason_bare}"
    print("Bug #2 PASSED: Markdown fences stripped and pass safety check")


def test_bug3_json_parsing():
    json_with_fences = '```json\n{"ablation_scripts": [{"name": "baseline", "code": "import numpy"}]}\n```'
    parsed = parse_json_response(json_with_fences)
    assert "ablation_scripts" in parsed
    assert len(parsed["ablation_scripts"]) == 1
    assert parsed["ablation_scripts"][0]["name"] == "baseline"

    json_no_newline = '```json{"key": "value"}```'
    parsed2 = parse_json_response(json_no_newline)
    assert parsed2 == {"key": "value"}
    print("Bug #3 PASSED: JSON parsing handles markdown fences")


def test_bug4_failure_score():
    fs_min = failure_score("minimize")
    fs_max = failure_score("maximize")
    assert fs_min == float("inf")
    assert fs_max == 0.0
    assert normalize_score(fs_min, "minimize") < normalize_score(0.05, "minimize")
    assert normalize_score(fs_max, "maximize") < normalize_score(0.50, "maximize")
    print("Bug #4 PASSED: failure_score is direction-aware")


def test_bug4_eval_candidate_fallback():
    # In real mode (mock off), empty code should get failure_score
    # In mock mode, eval_candidate returns random scores, so we test
    # the real-mode path by patching _is_mock_mode
    from unittest.mock import patch

    with patch(
        "src.mle_star.subgraphs.candidate_subgraph._is_mock_mode", return_value=False
    ):
        state_min = CandidateState(code="", score=0.0, metric_direction="minimize")
        result_min = eval_candidate(dict(state_min))
        assert result_min["score"] == float("inf"), (
            f"Empty code should get inf for minimize, got {result_min['score']}"
        )

        state_max = CandidateState(code="", score=0.0, metric_direction="maximize")
        result_max = eval_candidate(dict(state_max))
        assert result_max["score"] == 0.0, (
            f"Empty code should get 0.0 for maximize, got {result_max['score']}"
        )
    print("Bug #4 PASSED: eval_candidate uses failure_score correctly")


def test_cli_args():
    import argparse
    from main import _build_config

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
    assert config.max_outer_steps == 2
    assert config.max_inner_steps == 2
    assert config.max_ensemble_rounds == 2
    assert config.max_debug_retries == 2
    print("CLI args PASSED: Config propagates correctly")


if __name__ == "__main__":
    test_bug1_timeout_config()
    test_bug2_fence_stripping()
    test_bug3_json_parsing()
    test_bug4_failure_score()
    test_bug4_eval_candidate_fallback()
    test_cli_args()
    print("\n=== ALL INTEGRATION VERIFICATION TESTS PASSED ===")
