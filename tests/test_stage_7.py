"""Stage 7 verification tests: Search Phase LLM Integration.

Validates LLM infrastructure, search cache, prompt templates, code parsing,
execution sandbox, and mock-aware node implementations.

Test IDs follow the Stage 7 verification checklist.
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.mle_star.state.shared import (
    LLMConfig,
    call_llm,
    parse_score,
    parse_json_response,
    parse_code_block,
    replace_code_block,
    fuzzy_replace,
    parse_code_blocks,
    infer_metric_direction,
    normalize_score,
    display_score,
    _default_llm_config,
    _build_fallback_config,
    _call_chat_ollama,
)
from src.mle_star.config import MOCK_MODE, EXECUTION_TIMEOUT, NUM_RETRIEVED_MODELS
from src.mle_star.nodes.execution import validate_code_safety, execute_code
from src.mle_star.search_cache import SearchCache
from src.mle_star.prompts.search import (
    RETRIEVER_PROMPT,
    CANDIDATE_EVAL_PROMPT,
    MERGER_PROMPT,
)
from src.mle_star.prompts.robustness import (
    DATA_USAGE_CHECK_PROMPT,
    DATA_USAGE_FIX_PROMPT,
    DATA_LEAKAGE_EXTRACT_PROMPT,
    DATA_LEAKAGE_CORRECT_PROMPT,
    DEBUG_PROMPT,
)
from src.mle_star.state.alg1_state import (
    Alg1State,
    CandidateState,
    MergeState,
    FanoutState,
    MergeFanoutState,
)


# ── LLM Infrastructure Tests ──────────────────────────────────────────────


class TestLLMConfig:
    def test_s7_01_default_config(self):
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "glm-5.1:cloud"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.timeout == 300

    def test_s7_02_custom_config(self):
        config = LLMConfig(provider="openrouter", model="test-model", api_key="key123")
        assert config.provider == "openrouter"
        assert config.model == "test-model"
        assert config.api_key == "key123"

    def test_s7_03_default_llm_config_from_env(self):
        config = _default_llm_config()
        assert config.provider in ("ollama", "openrouter", "openai")
        assert config.model  # model should be non-empty

    def test_s7_04_base_url_defaults(self):
        config_ollama = LLMConfig(provider="ollama")
        assert "ollama.com" in config_ollama.base_url

        config_openrouter = LLMConfig(provider="openrouter")
        assert "openrouter" in config_openrouter.base_url


class TestParseScore:
    def test_s7_05_final_validation_performance(self):
        assert parse_score("Final Validation Performance: 0.4567") == 0.4567

    def test_s7_06_final_score(self):
        assert parse_score("Final Score: 0.1234") == 0.1234

    def test_s7_07_validation_score(self):
        assert parse_score("Validation Score: 0.9876") == 0.9876

    def test_s7_08_score_alone(self):
        assert parse_score("Score: 0.5555") == 0.5555

    def test_s7_09_fallback_last_number(self):
        assert parse_score("Some output 0.9999 more text") == 0.9999

    def test_s7_10_empty_output(self):
        assert parse_score("") is None

    def test_s7_11_no_numbers(self):
        assert parse_score("no score here") is None

    def test_s7_12_case_insensitive(self):
        assert parse_score("final validation performance: 0.5") == 0.5


class TestParseJsonResponse:
    def test_s7_13_direct_json(self):
        result = parse_json_response('{"a": 1, "b": 2}')
        assert result == {"a": 1, "b": 2}

    def test_s7_14_json_in_markdown(self):
        result = parse_json_response('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_s7_15_json_in_code_block(self):
        result = parse_json_response('```\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_s7_16_nested_json(self):
        result = parse_json_response('prefix {"models": [1, 2]} suffix')
        assert result == {"models": [1, 2]}

    def test_s7_17_empty_response(self):
        with pytest.raises(ValueError):
            parse_json_response("")


class TestParseCodeBlock:
    def test_s7_18_python_code_block(self):
        code = "```python\nimport pandas as pd\nprint('hello')\n```"
        result = parse_code_block(code)
        assert "import pandas" in result

    def test_s7_19_raw_code_block(self):
        code = "```\nimport numpy as np\n```"
        result = parse_code_block(code)
        assert "import numpy" in result

    def test_s7_20_code_without_fences(self):
        code = "import sklearn\nprint('hello')"
        result = parse_code_block(code)
        assert "import sklearn" in result

    def test_s7_21_no_code_raises(self):
        with pytest.raises(ValueError):
            parse_code_block("just some text without any code indicators")


class TestReplaceCodeBlock:
    def test_s7_22_exact_replacement(self):
        solution = "import numpy as np\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        result = replace_code_block(
            solution, "def foo():\n    return 1", "def foo():\n    return 42"
        )
        assert "return 42" in result

    def test_s7_23_ast_replacement(self):
        solution = "def foo():\n    return 1\n"
        result = replace_code_block(
            solution, "def foo():\n    return 1", "def foo():\n    return 42"
        )
        assert "return 42" in result

    def test_s7_24_fuzzy_replacement(self):
        solution = "def foo():\n    return 1\n\nSome other code here"
        result = fuzzy_replace(
            solution, "def foo():\n    return 1", "def foo():\n    return 42"
        )
        assert "return 42" in result

    def test_s7_25_empty_old_block(self):
        solution = "def foo():\n    return 1\n"
        result = replace_code_block(solution, "", "new code")
        assert result == solution


class TestParseCodeBlocks:
    def test_s7_26_function_blocks(self):
        solution = "import numpy as np\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        blocks = parse_code_blocks(solution)
        names = [b["name"] for b in blocks]
        assert "foo" in names
        assert "bar" in names

    def test_s7_27_assignment_blocks(self):
        solution = "X = 1\nY = 2\n"
        blocks = parse_code_blocks(solution)
        assert len(blocks) >= 2

    def test_s7_28_syntax_error_fallback(self):
        solution = "this is not valid python {{{"
        blocks = parse_code_blocks(solution)
        assert len(blocks) == 1
        assert blocks[0]["name"] == "full_module"


# ── Execution Sandbox Tests ──────────────────────────────────────────────


class TestValidateCodeSafety:
    def test_s7_29_safe_code(self):
        code = "import pandas as pd\nimport numpy as np\nprint('hello')\n"
        is_safe, reason = validate_code_safety(code)
        assert is_safe

    def test_s7_30_subprocess_import_blocked(self):
        code = "import subprocess\nsubprocess.run(['ls'])\n"
        is_safe, reason = validate_code_safety(code)
        assert not is_safe
        assert "subprocess" in reason

    def test_s7_31_eval_blocked(self):
        code = "x = eval('1+1')\n"
        is_safe, reason = validate_code_safety(code)
        assert not is_safe
        assert "eval" in reason

    def test_s7_32_empty_code(self):
        is_safe, reason = validate_code_safety("")
        assert not is_safe

    def test_s7_33_sklearn_imports_allowed(self):
        code = "from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor()\n"
        is_safe, reason = validate_code_safety(code)
        assert is_safe

    def test_s7_34_os_system_blocked(self):
        code = "import os\nos.system('rm -rf /')\n"
        is_safe, reason = validate_code_safety(code)
        assert not is_safe


class TestSearchCache:
    def test_s7_35_preseeded_cache(self):
        cache = SearchCache()
        models = cache._get_preseeded("nomad2018 sklearn models")
        assert len(models) > 0
        assert "model_name" in models[0]

    def test_s7_36_cache_path(self):
        cache = SearchCache(cache_dir="/tmp/test_cache")
        path = cache._cache_path("test query")
        assert path.endswith(".json")
        assert "/tmp/test_cache/" in path

    def test_s7_37_get_models_for_task(self):
        cache = SearchCache()
        models = cache.get_models_for_task("predict formation energy", num_models=3)
        assert len(models) <= 3
        assert all("model_name" in m for m in models)

    def test_s7_38_cache_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SearchCache(cache_dir=tmp)
            result = cache._save_cache("test query", [{"model_name": "TestModel"}])
            loaded = cache._load_cache("test query")
            assert loaded is not None
            assert loaded[0]["model_name"] == "TestModel"


# ── Prompt Template Tests ────────────────────────────────────────────────


class TestPromptTemplates:
    def test_s7_39_retriever_prompt_format(self):
        prompt = RETRIEVER_PROMPT.format(
            task_desc="predict house prices",
            metric="RMSLE",
            num_models=4,
        )
        assert "4" in prompt
        assert "sklearn" in prompt.lower()
        assert "predict house prices" in prompt

    def test_s7_40_candidate_eval_prompt_format(self):
        prompt = CANDIDATE_EVAL_PROMPT.format(
            task_desc="predict energy",
            model_name="RandomForestRegressor",
            model_description="Ensemble of decision trees",
            metric="RMSLE",
            direction="lower is better = minimize",
            feature_cols="spacegroup, number_of_total_atoms",
            target_cols="formation_energy_ev_natom, bandgap_energy_ev",
            additional_constraints="",
        )
        assert "RandomForestRegressor" in prompt
        assert "RMSLE" in prompt

    def test_s7_41_merger_prompt_format(self):
        prompt = MERGER_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            direction="lower is better = minimize",
            base_code="code1",
            ref_code="code2",
        )
        assert "code1" in prompt
        assert "code2" in prompt

    def test_s7_42_debug_prompt_format(self):
        prompt = DEBUG_PROMPT.format(
            task_desc="predict energy",
            metric="RMSLE",
            code="import numpy as np",
            error_message="ValueError",
            exit_code=1,
            stdout_output="",
        )
        assert "ValueError" in prompt
        assert "RMSLE" in prompt


# ── State Schema Tests ──────────────────────────────────────────────────


class TestAlg1StateUpdates:
    def test_s7_43_alg1_state_has_new_fields(self):
        state = Alg1State(
            task_desc="test",
            metric_direction="minimize",
            score_function_desc="RMSLE",
            retrieved_models=["RandomForestRegressor"],
            model_descriptions=[
                {"model_name": "RandomForestRegressor", "description": "test"}
            ],
            datasets=["input/train.csv"],
            candidates_pool=[],
            leaderboard=[],
            best_candidate={},
            current_reference_idx=0,
            stage_history=[],
            status="start",
        )
        assert state["score_function_desc"] == "RMSLE"
        assert state["model_descriptions"] == [
            {"model_name": "RandomForestRegressor", "description": "test"}
        ]
        assert state["datasets"] == ["input/train.csv"]

    def test_s7_44_candidate_state_has_new_fields(self):
        state = CandidateState(
            model="RandomForestRegressor",
            model_description={
                "model_name": "RandomForestRegressor",
                "description": "test",
            },
            task_desc="predict energy",
            score_function_desc="RMSLE",
            datasets=["input/train.csv"],
            metric_direction="minimize",
            code="import numpy as np",
            score=0.5,
            attempts=0,
            usage_fix_attempts=0,
            leakage_fix_attempts=0,
            execution_output="",
            execution_error=None,
            sub_events=[],
            status="pending",
        )
        assert state["model_description"] == {
            "model_name": "RandomForestRegressor",
            "description": "test",
        }
        assert state["task_desc"] == "predict energy"

    def test_s7_45_merge_state_has_new_fields(self):
        state = MergeState(
            base_code="code1",
            ref_code="code2",
            merged_code="",
            task_desc="predict energy",
            score_function_desc="RMSLE",
            datasets=["input/train.csv"],
            metric_direction="minimize",
            score=0.0,
            attempts=0,
            leakage_fix_attempts=0,
            execution_output="",
            execution_error=None,
            sub_events=[],
            status="pending",
        )
        assert state["task_desc"] == "predict energy"
        assert state["datasets"] == ["input/train.csv"]


# ── Mock Mode Tests ──────────────────────────────────────────────────────


class TestMockMode:
    def test_s7_46_config_mock_mode_default(self):
        os.environ.pop("MLE_MOCK_MODE", None)
        from src.mle_star.config import MOCK_MODE as _mock

        # Default should be False (since we didn't set env var)
        # But it could be True if previously set in this process
        assert isinstance(_mock, bool)

    def test_s7_47_candidate_subgraph_mock(self):
        from src.mle_star.subgraphs.candidate_subgraph import (
            A2__generate,
            A13__check_usage,
            eval_candidate,
        )

        os.environ["MLE_MOCK_MODE"] = "1"
        state = CandidateState(
            model="RandomForestRegressor",
            model_description=None,
            task_desc="test",
            score_function_desc="",
            datasets=[],
            metric_direction="minimize",
            code="",
            score=0.0,
            attempts=0,
            usage_fix_attempts=0,
            leakage_fix_attempts=0,
            execution_output="",
            execution_error=None,
            sub_events=[],
            status="pending",
        )
        result = A2__generate(dict(state))
        assert "code" in result
        assert "RandomForestRegressor" in result["code"] or result["code"].startswith(
            "code_"
        )
        os.environ.pop("MLE_MOCK_MODE", None)

    def test_s7_48_merge_subgraph_mock(self):
        from src.mle_star.subgraphs.merge_subgraph import A3__merge

        os.environ["MLE_MOCK_MODE"] = "1"
        state = MergeState(
            base_code="code1",
            ref_code="code2",
            merged_code="",
            task_desc="test",
            score_function_desc="",
            datasets=[],
            metric_direction="minimize",
            score=0.0,
            attempts=0,
            leakage_fix_attempts=0,
            execution_output="",
            execution_error=None,
            sub_events=[],
            status="pending",
        )
        result = A3__merge(dict(state))
        assert "merged_code" in result
        os.environ.pop("MLE_MOCK_MODE", None)


# ── Metric Direction Tests (extended) ────────────────────────────────────


class TestMetricDirectionExtended:
    def test_s7_49_rmsle_minimize(self):
        assert (
            infer_metric_direction("root mean squared logarithmic error") == "minimize"
        )

    def test_s7_50_mse_minimize(self):
        assert infer_metric_direction("mean squared error") == "minimize"

    def test_s7_51_accuracy_maximize(self):
        assert infer_metric_direction("classification accuracy") == "maximize"

    def test_s7_52_normalize_minimize(self):
        assert normalize_score(0.5, "minimize") == -0.5

    def test_s7_53_display_minimize(self):
        assert display_score(-0.5, "minimize") == 0.5


# ── ChatOllama Integration Tests ──────────────────────────────────────────


class TestChatOllamaRouting:
    def test_s7_54_ollama_base_url_is_api_ollama_com(self):
        config = LLMConfig(provider="ollama")
        assert config.base_url == "https://api.ollama.com"

    def test_s7_55_build_fallback_config_openrouter(self):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_MODEL_NAME"] = "test-model"
        config = _build_fallback_config("openrouter")
        assert config.provider == "openrouter"
        assert config.model == "test-model"
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("OPENROUTER_MODEL_NAME", None)

    def test_s7_56_build_fallback_config_ollama(self):
        config = _build_fallback_config("ollama")
        assert config.provider == "ollama"
        assert config.model != ""

    def test_s7_57_call_llm_routes_ollama_to_chat_ollama(self):
        with patch(
            "src.mle_star.state.shared._call_chat_ollama", return_value="test response"
        ) as mock_chat:
            config = LLMConfig(provider="ollama", model="glm-5.1:cloud")
            result = call_llm("test prompt", config=config)
            mock_chat.assert_called_once()
            assert result == "test response"

    def test_s7_58_call_llm_routes_openrouter_to_openai_compat(self):
        with patch(
            "src.mle_star.state.shared._call_openai_compatible",
            return_value="test response",
        ) as mock_httpx:
            config = LLMConfig(provider="openrouter", model="test-model", api_key="key")
            result = call_llm("test prompt", config=config)
            mock_httpx.assert_called_once()
            assert result == "test response"

    def test_s7_59_call_llm_fallback_on_primary_failure(self):
        with (
            patch(
                "src.mle_star.state.shared._call_chat_ollama",
                side_effect=Exception("Ollama down"),
            ),
            patch(
                "src.mle_star.state.shared._call_openai_compatible",
                return_value="fallback response",
            ) as mock_fallback,
        ):
            os.environ["OPENROUTER_API_KEY"] = "test-key"
            config = LLMConfig(provider="ollama", model="glm-5.1:cloud")
            result = call_llm("test prompt", config=config)
            mock_fallback.assert_called_once()
            assert result == "fallback response"
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_s7_60_call_llm_raises_when_no_fallback(self):
        with patch(
            "src.mle_star.state.shared._call_chat_ollama",
            side_effect=Exception("Ollama down"),
        ):
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("MLE_LLM_FALLBACK_PROVIDER", None)
            config = LLMConfig(provider="ollama", model="glm-5.1:cloud")
            with pytest.raises(Exception, match="Ollama down"):
                call_llm("test prompt", config=config)

    def test_s7_61_call_llm_no_self_fallback(self):
        with patch(
            "src.mle_star.state.shared._call_chat_ollama",
            side_effect=Exception("Ollama down"),
        ):
            os.environ["MLE_LLM_FALLBACK_PROVIDER"] = "ollama"
            config = LLMConfig(provider="ollama", model="glm-5.1:cloud")
            with pytest.raises(Exception, match="Ollama down"):
                call_llm("test prompt", config=config)
            os.environ.pop("MLE_LLM_FALLBACK_PROVIDER", None)


# ── llm_failed Routing Tests ─────────────────────────────────────────────


class TestLLMFailedRouting:
    def test_s7_62_candidate_llm_failed_skips_checks(self):
        from src.mle_star.subgraphs.candidate_subgraph import route_after_generate

        state = {"status": "llm_failed"}
        assert route_after_generate(state) == "__end__"

    def test_s7_63_candidate_llm_success_goes_to_check_usage(self):
        from src.mle_star.subgraphs.candidate_subgraph import route_after_generate

        state = {"status": "pending"}
        assert route_after_generate(state) == "A13__check_usage"

    def test_s7_64_merge_llm_failed_skips_checks(self):
        from src.mle_star.subgraphs.merge_subgraph import route_after_merge

        state = {"status": "llm_failed"}
        assert route_after_merge(state) == "__end__"

    def test_s7_65_merge_llm_success_goes_to_check_leakage(self):
        from src.mle_star.subgraphs.merge_subgraph import route_after_merge

        state = {"status": "pending"}
        assert route_after_merge(state) == "A12__check_leakage_merge"

    def test_s7_66_a2_generate_sets_llm_failed_on_error(self):
        with (
            patch(
                "src.mle_star.subgraphs.candidate_subgraph.call_llm",
                side_effect=Exception("LLM error"),
            ),
            patch(
                "src.mle_star.subgraphs.candidate_subgraph._is_mock_mode",
                return_value=False,
            ),
        ):
            from src.mle_star.subgraphs.candidate_subgraph import A2__generate

            state = CandidateState(
                model="TestModel",
                model_description=None,
                task_desc="test",
                score_function_desc="",
                datasets=[],
                metric_direction="minimize",
                code="",
                score=0.0,
                attempts=0,
                usage_fix_attempts=0,
                leakage_fix_attempts=0,
                execution_output="",
                execution_error=None,
                sub_events=[],
                status="pending",
            )
            result = A2__generate(dict(state))
            assert result["status"] == "llm_failed"
            assert result["score"] == float("inf")

    def test_s7_67_a3_merge_sets_llm_failed_on_error(self):
        with (
            patch(
                "src.mle_star.subgraphs.merge_subgraph.call_llm",
                side_effect=Exception("LLM error"),
            ),
            patch(
                "src.mle_star.subgraphs.merge_subgraph._is_mock_mode",
                return_value=False,
            ),
        ):
            from src.mle_star.subgraphs.merge_subgraph import A3__merge

            state = MergeState(
                base_code="code1",
                ref_code="code2",
                merged_code="",
                task_desc="test",
                score_function_desc="",
                datasets=[],
                metric_direction="minimize",
                score=0.0,
                attempts=0,
                leakage_fix_attempts=0,
                execution_output="",
                execution_error=None,
                sub_events=[],
                status="pending",
            )
            result = A3__merge(dict(state))
            assert result["status"] == "llm_failed"
            assert result["score"] == float("inf")
