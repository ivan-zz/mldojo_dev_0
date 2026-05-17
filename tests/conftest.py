"""Pytest configuration for MLE-STAR tests.

Disables simulated delays during testing for speed.
Patches logging to suppress stdout during e2e runs.
Enables MLE_MOCK_MODE to prevent real LLM calls during test execution.
"""

import logging
import os

import pytest

os.environ["MLE_MOCK_MODE"] = "1"


@pytest.fixture(autouse=True)
def _suppress_mle_stdout(caplog):
    """Suppress MLELogger stdout handler output during tests."""
    caplog.set_level(logging.DEBUG, logger="mle_star")


@pytest.fixture(scope="session")
def e2e_result():
    """Run Algorithm 1 once per test session and share the result."""
    from src.mle_star.algorithms.algorithm_1 import run

    import tempfile

    run_dir = tempfile.mkdtemp(prefix="mle_star_e2e_")
    initial_state = {
        "task_desc": "ML pipeline optimization",
        "retrieved_models": [],
        "candidates_pool": [],
        "leaderboard": [],
        "best_candidate": {},
        "current_reference_idx": 0,
        "stage_history": [],
        "status": "start",
    }
    return run(
        initial_state=initial_state,
        run_dir=run_dir,
        thread_id="e2e_session",
    )
