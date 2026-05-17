"""Stage 8 verification tests: Ablation + Refinement LLM Integration, Ensemble + Submission LLM Integration.

Validates prompt templates, mock/real dispatch, LLM call patterns, and
execution integration for all nodes modified in Stage 8.

Test IDs follow the s8_XX convention.
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from src.mle_star.config import MOCK_MODE


# ── Prompt Template Tests ─────────────────────────────────────────────────


class TestAblationPrompts:
    def test_s8_01_ablation_study_prompt_format(self):
        from src.mle_star.prompts.ablation import ABLATION_STUDY_PROMPT

        prompt = ABLATION_STUDY_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            direction="lower is better = minimize",
            solution="import numpy as np",
            functional_blocks="block1, block2",
            previous_summaries="none",
        )
        assert "predict energy" in prompt
        assert "RMSLE" in prompt
        assert "ablation_scripts" in prompt

    def test_s8_02_ablation_summarize_prompt_format(self):
        from src.mle_star.prompts.ablation import ABLATION_SUMMARIZE_PROMPT

        prompt = ABLATION_SUMMARIZE_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            ablation_results="baseline: 0.12, block1: 0.15",
        )
        assert "predict energy" in prompt
        assert "RMSLE" in prompt

    def test_s8_03_extractor_prompt_format(self):
        from src.mle_star.prompts.ablation import EXTRACTOR_PROMPT

        prompt = EXTRACTOR_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            ablation_summary="block1 is most impactful",
            solution="def foo(): pass",
            functional_blocks="block1, block2",
            previous_blocks="block0",
        )
        assert "predict energy" in prompt
        assert "block0" in prompt
        assert "target_block_name" in prompt


class TestRefinementPrompts:
    def test_s8_04_coder_prompt_format(self):
        from src.mle_star.prompts.refinement import CODER_PROMPT

        prompt = CODER_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            direction="lower is better = minimize",
            current_solution="import numpy",
            target_block="def foo(): pass",
            current_plan="improve feature engineering",
            previous_attempts="none",
        )
        assert "predict energy" in prompt
        assert "RMSLE" in prompt
        assert "foo" in prompt

    def test_s8_05_planner_prompt_format(self):
        from src.mle_star.prompts.refinement import PLANNER_PROMPT

        prompt = PLANNER_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            target_block="def foo(): pass",
            best_score="0.15",
            plan_history="plan1: 0.16",
            execution_output="Final Validation Performance: 0.15",
        )
        assert "predict energy" in prompt
        assert "0.15" in prompt


class TestVerificationPrompt:
    def test_s8_06_semantic_verify_prompt_format(self):
        from src.mle_star.prompts.verification import SEMANTIC_VERIFY_PROMPT

        prompt = SEMANTIC_VERIFY_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            current_solution="import numpy",
            refined_code="def foo(): return 42",
        )
        assert "predict energy" in prompt
        assert "semantic_fail" in prompt


class TestEnsemblePrompts:
    def test_s8_07_ensemble_planner_prompt_format(self):
        from src.mle_star.prompts.ensemble import ENSEMBLE_PLANNER_PROMPT

        prompt = ENSEMBLE_PLANNER_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            direction="lower is better = minimize",
            score_descriptions="Solution 1: 0.12",
            previous_plans="none",
            best_ensemble_score="0.12",
            ensemble_round="0",
        )
        assert "predict energy" in prompt
        assert "0.12" in prompt

    def test_s8_08_ensembler_prompt_format(self):
        from src.mle_star.prompts.ensemble import ENSEMBLER_PROMPT

        prompt = ENSEMBLER_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            direction="lower is better = minimize",
            plan="weight average",
            solutions_with_scores="Solution 1 (0.12): ...",
            best_ensemble_section="Previous best: ...",
        )
        assert "predict energy" in prompt
        assert "weight average" in prompt


class TestSubmissionPrompts:
    def test_s8_09_submission_prompt_format(self):
        from src.mle_star.prompts.submission import SUBMISSION_PROMPT

        prompt = SUBMISSION_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            solution="import numpy",
            best_score="0.12",
        )
        assert "predict energy" in prompt
        assert "submission.csv" in prompt

    def test_s8_10_subsampling_extract_prompt_format(self):
        from src.mle_star.prompts.submission import SUBSAMPLING_EXTRACT_PROMPT

        prompt = SUBSAMPLING_EXTRACT_PROMPT.format(code="df.sample(n=1000)")
        assert "has_subsampling" in prompt
        assert "sample" in prompt.lower() or "subsampling" in prompt.lower()

    def test_s8_11_subsampling_remove_prompt_format(self):
        from src.mle_star.prompts.submission import SUBSAMPLING_REMOVE_PROMPT

        prompt = SUBSAMPLING_REMOVE_PROMPT.format(
            code="df = df.sample(n=1000)",
            subsampling_block="df = df.sample(n=1000)",
        )
        assert "df.sample" in prompt


# ── Mock Mode Tests ───────────────────────────────────────────────────────


class TestMockModeAblation:
    def test_s8_12_a4_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.ablation import A4__generate_ablation

        state = {
            "current_solution": "def foo(): pass",
            "best_score": 0.85,
        }
        result = A4__generate_ablation(state)
        assert "ablation_scripts" in result
        assert "functional_blocks" in result
        assert result["status"] == "ablation_generated"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_13_a5_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.ablation import A5__summarize_ablation

        state = {"ablation_results_list": [{"block_name": "foo", "impact": 0.05}]}
        result = A5__summarize_ablation(state)
        assert "ablation_summaries" in result
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_14_a6_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.ablation import A6__extract_block

        state = {
            "ablation_results_list": [{"block_name": "foo", "impact": 0.05}],
            "functional_blocks": [],
            "previous_blocks": [],
        }
        result = A6__extract_block(state)
        assert "target_block" in result
        assert "initial_plan" in result
        os.environ.pop("MLE_MOCK_MODE", None)


class TestMockModeRefinement:
    def test_s8_15_a7_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.refinement import A7__implement

        state = {
            "current_plan": "improve features",
            "target_block": "def foo(): pass",
            "current_solution": "def foo(): pass\ndef bar(): return 1",
        }
        result = A7__implement(state)
        assert "refined_code" in result
        assert "candidate_solution" in result
        assert result["status"] == "implemented"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_16_a8_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.refinement import A8__plan

        state = {
            "target_block": "def foo(): pass",
            "inner_step": 0,
            "execution_score": 0.15,
        }
        result = A8__plan(state)
        assert "current_plan" in result
        assert result["inner_step"] == 1
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_17_a_verify_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.refinement import A_verify

        state = {"refined_code": "def foo(): pass"}
        result = A_verify(state)
        assert result["status"] == "ok"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_18_eval_refinement_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.refinement import eval_refinement

        state = {
            "candidate_solution": "print('hello')",
            "best_score": 0.85,
            "debug_retries": 0,
            "inner_step": 0,
        }
        result = eval_refinement(state)
        assert result["status"] in ("ok", "error")
        os.environ.pop("MLE_MOCK_MODE", None)


class TestMockModeEnsemble:
    def test_s8_19_a9_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.ensemble import A9__plan_ensemble

        state = {
            "ensemble_solutions": ["code1", "code2"],
            "ensemble_input_scores": [0.12, 0.15],
            "ensemble_round": 0,
        }
        result = A9__plan_ensemble(state)
        assert "current_ensemble_plan" in result
        assert result["status"] == "planned"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_20_a10_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.ensemble import A10__implement_ensemble

        state = {
            "current_ensemble_plan": "weighted average",
            "ensemble_solutions": ["code1", "code2"],
            "best_ensemble_code": "code1",
            "ensemble_round": 0,
        }
        result = A10__implement_ensemble(state)
        assert "current_ensemble_code" in result
        assert result["status"] == "implemented"
        os.environ.pop("MLE_MOCK_MODE", None)


class TestMockModeSubmission:
    def test_s8_21_submit_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.submission import A_test__submit

        state = {"final_solution": "import numpy as np"}
        result = A_test__submit(state)
        assert "submission_code" in result
        assert result["status"] == "generated"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_22_subsampling_extract_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.submission import subsampling_extract

        state = {"submission_code": "import numpy as np"}
        result = subsampling_extract(state)
        assert "subsampling_block" in result
        assert result["status"] == "extracted"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_23_subsampling_remove_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.submission import subsampling_remove

        state = {"submission_code": "import numpy as np\n# code"}
        result = subsampling_remove(state)
        assert "submission_code" in result
        assert result["status"] == "subsampling_removed"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_24_check_leakage_submission_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.submission import A12__check_leakage_submission

        state = {"submission_code": "import numpy"}
        result = A12__check_leakage_submission(state)
        assert result["leakage_status"] == "ok"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_25_fix_leakage_submission_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.submission import A12__fix_leakage_submission

        state = {"submission_code": "import numpy"}
        result = A12__fix_leakage_submission(state)
        assert "submission_code" in result
        assert result["status"] == "leakage_fix_applied"
        os.environ.pop("MLE_MOCK_MODE", None)


class TestMockModeRobustnessEnsemble:
    def test_s8_26_debug_ensemble_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.robustness import A11__debug_ensemble

        state = {
            "current_ensemble_code": "import numpy",
            "debug_retries": 0,
            "execution_error": "test error",
        }
        result = A11__debug_ensemble(state)
        assert result["debug_retries"] == 1
        assert result["status"] == "debugged"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_27_check_leakage_ensemble_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.robustness import A12__check_leakage_ensemble

        state = {"current_ensemble_code": "import numpy"}
        result = A12__check_leakage_ensemble(state)
        assert result["leakage_status"] == "ok"
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_28_fix_leakage_ensemble_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.robustness import A12__fix_leakage_ensemble

        state = {"current_ensemble_code": "import numpy"}
        result = A12__fix_leakage_ensemble(state)
        assert "current_ensemble_code" in result
        assert result["status"] == "leakage_fix_applied"
        os.environ.pop("MLE_MOCK_MODE", None)


# ── LLM Dispatch Tests (with mock LLM) ──────────────────────────────────────


class TestAblationRealMode:
    def test_s8_29_a4_real_mode_llm_success(self):
        with patch("src.mle_star.nodes.ablation._is_mock_mode", return_value=False):
            from src.mle_star.nodes.ablation import A4__generate_ablation

            mock_response = json.dumps(
                {
                    "ablation_scripts": [
                        {
                            "name": "baseline",
                            "block_name": "baseline",
                            "code": "print(1)",
                        },
                        {
                            "name": "ablation_foo",
                            "block_name": "foo",
                            "code": "print(2)",
                        },
                    ]
                }
            )
            with patch(
                "src.mle_star.nodes.ablation.call_llm", return_value=mock_response
            ):
                state = {
                    "current_solution": "def foo(): pass\nprint(foo())",
                    "best_score": 0.85,
                    "metric_direction": "minimize",
                    "ablation_summaries": [],
                }
                result = A4__generate_ablation(state)
                assert result["status"] == "ablation_generated"
                assert len(result["ablation_scripts"]) >= 2

    def test_s8_30_a4_real_mode_llm_failure_fallback(self):
        with patch("src.mle_star.nodes.ablation._is_mock_mode", return_value=False):
            from src.mle_star.nodes.ablation import A4__generate_ablation

            with patch(
                "src.mle_star.nodes.ablation.call_llm",
                side_effect=Exception("LLM down"),
            ):
                state = {
                    "current_solution": "def foo(): pass\nprint(foo())",
                    "best_score": 0.85,
                    "metric_direction": "minimize",
                    "ablation_summaries": [],
                }
                result = A4__generate_ablation(state)
                assert result["status"] == "ablation_generated"
                assert len(result["ablation_scripts"]) > 0


class TestRefinementRealMode:
    def test_s8_31_a7_real_mode_llm_success(self):
        with patch("src.mle_star.nodes.refinement._is_mock_mode", return_value=False):
            from src.mle_star.nodes.refinement import A7__implement

            with patch(
                "src.mle_star.nodes.refinement.call_llm",
                return_value="```python\ndef improved(): return 42\n```",
            ):
                state = {
                    "current_plan": "improve features",
                    "target_block": "def foo(): pass",
                    "current_solution": "def foo(): pass\nx = 1",
                    "best_score": 0.15,
                    "metric_direction": "minimize",
                    "debug_retries": 0,
                    "inner_step": 0,
                }
                result = A7__implement(state)
                assert result["status"] == "implemented"
                assert "refined_code" in result

    def test_s8_32_a8_real_mode_llm_success(self):
        with patch("src.mle_star.nodes.refinement._is_mock_mode", return_value=False):
            from src.mle_star.nodes.refinement import A8__plan

            with patch(
                "src.mle_star.nodes.refinement.call_llm",
                return_value="Try XGBoost for better feature interactions",
            ):
                state = {
                    "target_block": "def foo(): pass",
                    "best_score": 0.15,
                    "metric_direction": "minimize",
                    "execution_output": "Score: 0.15",
                    "current_plan": "",
                    "current_plans": [],
                    "current_scores": [],
                    "inner_step": 0,
                }
                result = A8__plan(state)
                assert "current_plan" in result
                assert result["inner_step"] == 1


class TestEnsembleRealMode:
    def test_s8_33_a9_real_mode_llm_success(self):
        with patch("src.mle_star.nodes.ensemble._is_mock_mode", return_value=False):
            from src.mle_star.nodes.ensemble import A9__plan_ensemble

            with patch(
                "src.mle_star.nodes.ensemble.call_llm",
                return_value="Weighted average of 2 models",
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

    def test_s8_34_a10_real_mode_llm_success(self):
        with patch("src.mle_star.nodes.ensemble._is_mock_mode", return_value=False):
            from src.mle_star.nodes.ensemble import A10__implement_ensemble

            with patch(
                "src.mle_star.nodes.ensemble.call_llm",
                return_value="```python\nimport numpy as np\nprint('ensemble')\n```",
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
                assert result["status"] == "implemented"
                assert "current_ensemble_code" in result


class TestSubmissionRealMode:
    def test_s8_35_submit_real_mode_llm_success(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import A_test__submit

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                return_value="```python\nimport pandas as pd\nprint('submission')\n```",
            ):
                state = {
                    "final_solution": "old code",
                    "best_solution": "old code",
                    "best_score": 0.12,
                    "task_desc": "predict energy",
                    "metric_direction": "minimize",
                }
                result = A_test__submit(state)
                assert result["status"] == "generated"
                assert "submission_code" in result

    def test_s8_36_subsampling_extract_real_mode_llm(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import subsampling_extract

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                return_value=json.dumps(
                    {
                        "has_subsampling": True,
                        "subsampling_block": "df = df.sample(n=1000)",
                    }
                ),
            ):
                state = {"submission_code": "df = df.sample(n=1000)\nprint(df.shape)"}
                result = subsampling_extract(state)
                assert "subsampling_block" in result

    def test_s8_37_check_leakage_submission_real_mode(self):
        with patch("src.mle_star.nodes.submission._is_mock_mode", return_value=False):
            from src.mle_star.nodes.submission import A12__check_leakage_submission

            with patch(
                "src.mle_star.nodes.submission.call_llm",
                return_value=json.dumps(
                    {
                        "has_leakage": False,
                        "leakage_issues": [],
                        "explanation": "No leakage",
                    }
                ),
            ):
                state = {
                    "submission_code": "import numpy",
                    "task_desc": "predict",
                    "score_function_desc": "RMSLE",
                }
                result = A12__check_leakage_submission(state)
                assert result["leakage_status"] in ("ok", "Yes Data Leakage")


class TestAblationVariantRealMode:
    def test_s8_38_eval_ablation_variant_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.subgraphs.ablation_variant_subgraph import (
            eval_ablation_variant,
        )

        state = {
            "variant_name": "baseline",
            "variant_code": "print(1)",
            "block_name": "baseline",
            "attempts": 0,
        }
        result = eval_ablation_variant(state)
        assert result["status"] in ("ok", "error")
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_39_debug_ablation_variant_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.subgraphs.ablation_variant_subgraph import (
            A11__debug_ablation_variant,
        )

        state = {"variant_code": "print(1)", "attempts": 0, "execution_error": "test"}
        result = A11__debug_ablation_variant(state)
        assert result["attempts"] == 1
        assert result["status"] == "ok"
        os.environ.pop("MLE_MOCK_MODE", None)


class TestEvalEnsembleSubmission:
    def test_s8_40_eval_ensemble_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.execution import eval_ensemble

        state = {
            "current_ensemble_code": "print(1)",
            "best_ensemble_score": 0.12,
            "ensemble_round": 0,
            "metric_direction": "minimize",
            "debug_retries": 0,
        }
        result = eval_ensemble(state)
        assert result["status"] in ("ok", "error")
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s8_41_eval_submission_mock_mode(self):
        os.environ["MLE_MOCK_MODE"] = "1"
        from src.mle_star.nodes.execution import eval_submission

        state = {
            "submission_code": "print(1)",
            "best_score": 0.12,
            "metric_direction": "minimize",
        }
        result = eval_submission(state)
        assert "submission_score" in result
        os.environ.pop("MLE_MOCK_MODE", None)


class TestRoutingFunctions:
    def test_s8_42_route_after_verify_semantic_fail(self):
        from src.mle_star.nodes.refinement import route_after_verify

        assert route_after_verify({"status": "semantic_fail"}) == "A7__implement"
        assert route_after_verify({"status": "ok"}) == "A_sast"

    def test_s8_43_route_after_sast_critical(self):
        from src.mle_star.nodes.refinement import route_after_sast

        assert route_after_sast({"status": "critical_violation"}) == "A7__implement"
        assert route_after_sast({"status": "pass"}) == "eval_refinement"

    def test_s8_44_route_after_eval_step(self):
        from langgraph.graph import END
        from src.mle_star.nodes.refinement import route_after_eval_step

        assert route_after_eval_step({"status": "ok"}) == END
        assert (
            route_after_eval_step({"status": "error", "debug_retries": 0})
            == "A11__debug_refine"
        )
        assert route_after_eval_step({"status": "error", "debug_retries": 3}) == END

    def test_s8_45_route_after_leakage_check_submission(self):
        from src.mle_star.nodes.submission import route_after_leakage_check_submission

        assert (
            route_after_leakage_check_submission({"leakage_status": "Yes Data Leakage"})
            == "A12__fix_leakage_submission"
        )
        assert (
            route_after_leakage_check_submission({"leakage_status": "ok"})
            == "eval_submission"
        )

    def test_s8_46_route_after_leakage_check_ensemble(self):
        from src.mle_star.nodes.robustness import route_after_leakage_check_ensemble

        assert (
            route_after_leakage_check_ensemble({"leakage_status": "Yes Data Leakage"})
            == "A12__fix_leakage_ensemble"
        )
        assert (
            route_after_leakage_check_ensemble({"leakage_status": "ok"})
            == "eval_ensemble"
        )

    def test_s8_47_route_after_ensemble_eval(self):
        from langgraph.graph import END
        from src.mle_star.nodes.robustness import route_after_ensemble_eval

        assert route_after_ensemble_eval({"status": "ok"}) == END
        assert (
            route_after_ensemble_eval({"status": "error", "debug_retries": 0})
            == "A11__debug_ensemble"
        )
        assert route_after_ensemble_eval({"status": "error", "debug_retries": 3}) == END


class TestParseCodeBlocksIntegration:
    def test_s8_48_parse_code_blocks_function_defs(self):
        from src.mle_star.state.shared import parse_code_blocks

        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        blocks = parse_code_blocks(code)
        names = [b["name"] for b in blocks]
        assert "foo" in names
        assert "bar" in names

    def test_s8_49_replace_code_block_basic(self):
        from src.mle_star.state.shared import replace_code_block

        solution = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        result = replace_code_block(
            solution, "def foo():\n    return 1", "def foo():\n    return 42"
        )
        assert "return 42" in result
        assert "return 2" in result
