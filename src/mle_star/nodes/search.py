"""Search phase nodes: A1__retrieve.

A1__retrieve uses SearchCache + Tavily API (or LLM) to find relevant ML models.
The actual A2__generate, A3__merge, and robustness nodes live in
candidate_subgraph.py and merge_subgraph.py since they are compiled into
subgraphs with conditional edges.
"""

import os

from src.mle_star.config import MOCK_MODE, NUM_RETRIEVED_MODELS
from src.mle_star.search_cache import SearchCache
from src.mle_star.state.shared import (
    call_llm,
    log_node_event,
    parse_json_response,
    traceable,
    _default_llm_config,
    format_direction,
)
from src.mle_star.prompts.search import RETRIEVER_PROMPT


@traceable("A1__retrieve")
def A1__retrieve(state: dict) -> dict:
    """Retrieve model candidates for evaluation.

    A1 from MLE-STAR paper - retrieves a list of model configurations
    to generate initial solutions from. Uses SearchCache for local caching
    and LLM for model suggestions.

    In mock mode, returns hardcoded default models.
    """
    if MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in ("1", "true", "yes"):
        return {
            "retrieved_models": [
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "Ridge",
                "SVR",
            ],
            "model_descriptions": [],
        }

    task_desc = state.get("task_desc", "ML regression task")
    metric_direction = state.get("metric_direction", "minimize")
    score_function_desc = state.get("score_function_desc", "")
    num_models = state.get("num_retrieved_models") or NUM_RETRIEVED_MODELS

    metric = score_function_desc or (
        "RMSLE" if metric_direction == "minimize" else "accuracy"
    )

    cache = SearchCache()

    cached_models = cache.get_models_for_task(task_desc, num_models=num_models)
    if cached_models and len(cached_models) >= num_models:
        model_names = [m["model_name"] for m in cached_models[:num_models]]
        log_node_event(
            "A1__retrieve",
            "cache_hit",
            {"num_models": len(cached_models), "models": model_names},
        )
        return {
            "retrieved_models": model_names,
            "model_descriptions": cached_models[:num_models],
        }

    prompt = RETRIEVER_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        num_models=num_models,
    )

    try:
        config = _default_llm_config()
        response = call_llm(prompt, response_format="json", config=config)
        parsed = parse_json_response(response)

        models = parsed.get("models", [])
        model_descriptions = []
        model_names = []

        for m in models:
            name = m.get("model_name", "")
            if name and name not in model_names:
                model_names.append(name)
                model_descriptions.append(
                    {
                        "model_name": name,
                        "description": m.get("description", ""),
                        "example_code": m.get("example_code", ""),
                        "category": m.get("category", "llm_suggested"),
                        "sklearn_module": m.get("sklearn_module", "sklearn"),
                    }
                )

        if not model_names:
            fallback = [
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "Ridge",
                "SVR",
            ]
            model_names = fallback[:num_models]
            model_descriptions = [
                {
                    "model_name": n,
                    "description": f"Default {n}",
                    "example_code": "",
                    "category": "fallback",
                    "sklearn_module": "sklearn",
                }
                for n in model_names
            ]

        log_node_event(
            "A1__retrieve",
            "llm_result",
            {"num_models": len(model_names), "models": model_names},
        )

        return {
            "retrieved_models": model_names[:num_models],
            "model_descriptions": model_descriptions[:num_models],
        }

    except Exception as e:
        log_node_event(
            "A1__retrieve",
            "error",
            {"error": str(e)[:200]},
        )
        fallback = [
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "Ridge",
            "SVR",
        ]
        return {
            "retrieved_models": fallback[:num_models],
            "model_descriptions": [
                {
                    "model_name": n,
                    "description": f"Fallback {n}",
                    "example_code": "",
                    "category": "fallback",
                    "sklearn_module": "sklearn",
                }
                for n in fallback[:num_models]
            ],
        }
