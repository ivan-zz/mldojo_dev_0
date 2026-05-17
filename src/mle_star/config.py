"""Configuration for MLE-STAR hyperparameters.

Reads from environment variables with sensible defaults.
All configurable hyperparameters are centralized here.
"""

import os


def _env_int(key: str, default: int) -> int:
    """Read an integer from environment variable, with fallback default."""
    val = os.environ.get(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _env_float(key: str, default: float) -> float:
    """Read a float from environment variable, with fallback default."""
    val = os.environ.get(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


# ── Algorithm 2: Outer loop (ablation) ─────────────────────────────────

MAX_OUTER_STEPS = _env_int("MLE_MAX_OUTER_STEPS", 4)

# ── Algorithm 2: Inner loop (refinement) ────────────────────────────────

MAX_INNER_STEPS = _env_int("MLE_MAX_INNER_STEPS", 4)

# ── Debug retry limits ──────────────────────────────────────────────────

MAX_DEBUG_RETRIES = _env_int("MLE_MAX_DEBUG_RETRIES", 3)

# ── Ablation variant execution ───────────────────────────────────────────

MAX_ABLATION_DEBUG_RETRIES = _env_int("MLE_MAX_ABLATION_DEBUG_RETRIES", 3)

# ── Parallel pipeline fan-out (D15) ─────────────────────────────────────

NUM_PARALLEL_SOLUTIONS = _env_int("MLE_NUM_PARALLEL_SOLUTIONS", 2)

# ── Algorithm 3: Ensemble rounds ────────────────────────────────────────

MAX_ENSEMBLE_ROUNDS = _env_int("MLE_MAX_ENSEMBLE_ROUNDS", 5)

# ── Ensemble debug retry limit ──────────────────────────────────────────

MAX_ENSEMBLE_DEBUG_RETRIES = _env_int("MLE_MAX_ENSEMBLE_DEBUG_RETRIES", 3)

# ── Leakage fix retry limit ─────────────────────────────────────────────

MAX_LEAKAGE_FIX_RETRIES = _env_int("MLE_MAX_LEAKAGE_FIX_RETRIES", 5)

# ── Subsampling threshold ────────────────────────────────────────────────

SUBSAMPLING_THRESHOLD = _env_int("MLE_SUBSAMPLING_THRESHOLD", 30000)

# ── Execution ────────────────────────────────────────────────────────────

EXECUTION_TIMEOUT = _env_int("MLE_EXECUTION_TIMEOUT", 900)

EXECUTION_MAX_MEMORY_MB = _env_int("MLE_EXECUTION_MAX_MEMORY_MB", 4096)

EXECUTION_MAX_CPU_SECONDS = _env_int("MLE_EXECUTION_MAX_CPU_SECONDS", 1800)

# ── Mock mode ────────────────────────────────────────────────────────────

MOCK_MODE = os.environ.get("MLE_MOCK_MODE", "").lower() in ("1", "true", "yes")

# ── LLM call limits (Stage 10) ────────────────────────────────────────────

MAX_LLM_CALLS_PER_PHASE = _env_int("MLE_MAX_LLM_CALLS_PER_PHASE", 100)

# ── Per-node timeout (Stage 10) ────────────────────────────────────────────

PER_NODE_TIMEOUT_SECONDS = _env_int("MLE_PER_NODE_TIMEOUT_SECONDS", 600)

# ── Docker sandbox (Stage 10) ─────────────────────────────────────────────

DOCKER_SANDBOX_ENABLED = os.environ.get("MLE_DOCKER_SANDBOX", "").lower() in (
    "1",
    "true",
    "yes",
)

DOCKER_IMAGE = os.environ.get(
    "MLE_DOCKER_IMAGE", "ghcr.io/all-hands-ai/od_runtime:latest"
)

DOCKER_CPU_LIMIT = _env_int("MLE_DOCKER_CPU_LIMIT", 1)

DOCKER_MEMORY_LIMIT_MB = _env_int("MLE_DOCKER_MEMORY_LIMIT_MB", 4096)

# ── Search ──────────────────────────────────────────────────────────────────

NUM_RETRIEVED_MODELS = _env_int("MLE_NUM_RETRIEVED_MODELS", 4)
