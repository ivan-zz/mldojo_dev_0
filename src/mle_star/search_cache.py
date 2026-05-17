"""Search result cache with Tavily API integration.

Minimizes Tavily search calls by caching results locally in JSON files.
Supports pre-seeded cache files for common tasks (e.g., sklearn models for
Nomad2018). When MLE_MOCK_MODE is enabled, returns cached results only
without making any API calls.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "input",
    "search_cache",
)

_MOCK_MODE = os.environ.get("MLE_MOCK_MODE", "").lower() in ("1", "true", "yes")


class SearchCache:
    """Local JSON file cache for web search results.

    Caches search results in {cache_dir}/{query_hash}.json files.
    Pre-seeded data files (e.g., nomad2018_sklearn_models.json) provide
    default results for common queries without any API calls.

    Args:
        cache_dir: Directory for cache files. Defaults to input/search_cache/.
        force_refresh: If True, always make API calls and update cache.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        force_refresh: bool = False,
    ):
        self.cache_dir = cache_dir or _CACHE_DIR
        self.force_refresh = force_refresh
        os.makedirs(self.cache_dir, exist_ok=True)
        self._preseeded: Optional[List[Dict]] = None

    def _cache_path(self, query: str) -> str:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{query_hash}.json")

    def search(
        self,
        query: str,
        max_results: int = 10,
        force_refresh: Optional[bool] = None,
    ) -> List[Dict]:
        """Search for results, using cache when available.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            force_refresh: Override instance force_refresh for this call.

        Returns:
            List of search result dicts with 'model_name', 'description',
            'example_code' keys.
        """
        should_refresh = (
            force_refresh if force_refresh is not None else self.force_refresh
        )

        if not should_refresh:
            cached = self._load_cache(query)
            if cached is not None:
                return cached[:max_results]

        if _MOCK_MODE:
            return self._get_preseeded(query)[:max_results]

        results = self._tavily_search(query, max_results)
        if results:
            self._save_cache(query, results)

        return results[:max_results]

    def get_models_for_task(
        self,
        task_desc: str,
        num_models: int = 4,
        constraint: str = "sklearn only",
    ) -> List[Dict]:
        """Get ML model suggestions for a task description.

        Checks pre-seeded cache first, then falls back to Tavily search.

        Args:
            task_desc: Task description string.
            num_models: Number of models to return.
            constraint: Model constraint string (e.g., "sklearn only").

        Returns:
            List of model dicts with 'model_name', 'description', 'example_code'.
        """
        preseeded = self._get_preseeded(task_desc)
        if preseeded and len(preseeded) >= num_models:
            import random as _random

            _random.seed(hash(task_desc) % 2**32)
            selected = _random.sample(preseeded, min(num_models, len(preseeded)))
            return selected

        query = (
            f"best {constraint} machine learning models for regression on tabular data. "
            f"Task: {task_desc[:200]} "
            f"Only scikit-learn models. Models must be from sklearn library."
        )
        results = self.search(query, max_results=num_models * 2)
        models = []
        for r in results:
            if "model_name" in r:
                models.append(r)
            elif "title" in r and "content" in r:
                models.append(
                    {
                        "model_name": r.get("title", "Unknown"),
                        "description": r.get("content", "")[:300],
                        "example_code": "",
                        "category": "search_result",
                        "sklearn_module": "sklearn",
                    }
                )
        return models[:num_models]

    def _get_preseeded(self, task_desc: str) -> List[Dict]:
        """Load and return pre-seeded model suggestions.

        Searches all JSON files in the cache directory for pre-seeded data.
        The first file with valid model entries is used.
        """
        if self._preseeded is not None:
            return self._preseeded

        for fname in sorted(os.listdir(self.cache_dir)):
            if fname.endswith(".json") and "sklearn" in fname.lower():
                fpath = os.path.join(self.cache_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    if (
                        isinstance(data, list)
                        and len(data) > 0
                        and "model_name" in data[0]
                    ):
                        self._preseeded = data
                        return data
                except (json.JSONDecodeError, OSError):
                    continue

        return []

    def _load_cache(self, query: str) -> Optional[List[Dict]]:
        """Load cached search results for a query."""
        cache_path = self._cache_path(query)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _save_cache(self, query: str, results: List[Dict]) -> None:
        """Save search results to cache."""
        cache_path = self._cache_path(query)
        try:
            with open(cache_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _tavily_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Perform a Tavily web search and return structured results.

        Returns:
            List of dicts with 'model_name', 'description', 'example_code',
            'category', 'sklearn_module' keys extracted from search results.
        """
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return []

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            response = client.search(query, max_results=max_results)

            results = []
            for item in response.get("results", []):
                title = item.get("title", "")
                content = item.get("content", "")
                url = item.get("url", "")

                model_name = _extract_model_name(title, content)
                if model_name:
                    results.append(
                        {
                            "model_name": model_name,
                            "description": content[:300],
                            "example_code": "",
                            "category": "search_result",
                            "sklearn_module": _infer_sklearn_module(model_name),
                            "source_url": url,
                        }
                    )

            return results

        except Exception as e:
            from src.mle_star.state.shared import log_node_event

            log_node_event(
                "SearchCache",
                "tavily_error",
                {"error": str(e)[:200], "query": query[:100]},
            )
            return []


_SKLEARN_MODEL_MODULES = {
    "RandomForestRegressor": "sklearn.ensemble",
    "RandomForestClassifier": "sklearn.ensemble",
    "GradientBoostingRegressor": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "AdaBoostRegressor": "sklearn.ensemble",
    "AdaBoostClassifier": "sklearn.ensemble",
    "BaggingRegressor": "sklearn.ensemble",
    "BaggingClassifier": "sklearn.ensemble",
    "ExtraTreesRegressor": "sklearn.ensemble",
    "ExtraTreesClassifier": "sklearn.ensemble",
    "Ridge": "sklearn.linear_model",
    "Lasso": "sklearn.linear_model",
    "ElasticNet": "sklearn.linear_model",
    "BayesianRidge": "sklearn.linear_model",
    "HuberRegressor": "sklearn.linear_model",
    "SVR": "sklearn.svm",
    "SVC": "sklearn.svm",
    "LinearSVR": "sklearn.svm",
    "KNeighborsRegressor": "sklearn.neighbors",
    "KNeighborsClassifier": "sklearn.neighbors",
    "MLPRegressor": "sklearn.neural_network",
    "MLPClassifier": "sklearn.neural_network",
    "DecisionTreeRegressor": "sklearn.tree",
    "DecisionTreeClassifier": "sklearn.tree",
}


def _extract_model_name(title: str, content: str) -> str:
    """Extract a sklearn model name from search result title/content."""
    for name in _SKLEARN_MODEL_MODULES:
        if name in title or name in content:
            return name
    return ""


def _infer_sklearn_module(model_name: str) -> str:
    """Infer the sklearn module for a model name."""
    return _SKLEARN_MODEL_MODULES.get(model_name, "sklearn")
