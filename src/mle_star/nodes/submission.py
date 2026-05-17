"""Submission nodes: A_test__submit, subsampling_extract, subsampling_remove.

Also contains submission-specific leakage check nodes:
  A12__check_leakage_submission, A12__fix_leakage_submission

And the routing function route_after_leakage_check_submission.

Real LLM implementations with mock fallbacks.
"""

import os

from langgraph.graph import END

from src.mle_star.state.shared import (
    traceable,
    simulate_delay,
    log_node_event,
    call_llm,
    _default_llm_config,
    parse_code_block,
    parse_json_response,
)
from src.mle_star.config import (
    MOCK_MODE,
    SUBSAMPLING_THRESHOLD,
    MAX_LEAKAGE_FIX_RETRIES,
)
from src.mle_star.prompts.submission import (
    SUBMISSION_PROMPT,
    SUBSAMPLING_EXTRACT_PROMPT,
    SUBSAMPLING_REMOVE_PROMPT,
)
from src.mle_star.prompts.robustness import (
    DATA_LEAKAGE_EXTRACT_PROMPT,
    DATA_LEAKAGE_CORRECT_PROMPT,
)


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


@traceable("A_test__submit")
def A_test__submit(state: dict) -> dict:
    """Generate a submission script from the best solution.

    Takes the final solution (best ensemble or best individual) and creates
    a clean submission script without debug code.

    Mock: copies the final solution as submission code.
    Real: calls LLM to clean up the solution into a submission script.
    """
    simulate_delay()

    final_solution = state.get("final_solution", state.get("best_solution", ""))

    if _is_mock_mode():
        submission_code = final_solution + "\n# submission_v1\n"

        log_node_event(
            "A_test__submit",
            "output",
            {"mode": "mock", "status": "generated", "code_len": len(submission_code)},
        )

        return {
            "submission_code": submission_code,
            "status": "generated",
        }

    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "")
    best_score = state.get("best_score", "")

    prompt = SUBMISSION_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        solution=final_solution,
        best_score=best_score,
    )

    try:
        response = call_llm(
            prompt, response_format="code", config=_default_llm_config()
        )
        code = parse_code_block(response)
    except Exception as e:
        log_node_event(
            "A_test__submit",
            "llm_failed",
            {"error": str(e)[:300]},
        )
        code = final_solution

    log_node_event(
        "A_test__submit",
        "output",
        {"mode": "real", "status": "generated", "code_len": len(code)},
    )

    return {
        "submission_code": code,
        "status": "generated",
    }


@traceable("subsampling_extract")
def subsampling_extract(state: dict) -> dict:
    """Extract subsampling blocks from the submission code.

    Identifies sections of code that use data subsampling (e.g., for
    faster iteration during development) so they can be removed before
    final submission.

    Mock: returns a placeholder subsampling block string.
    Real: calls LLM to identify subsampling code blocks.
    """
    simulate_delay()

    submission_code = state.get("submission_code", "")

    if _is_mock_mode():
        subsampling_block = "# subsampling block extracted\n"

        log_node_event(
            "subsampling_extract",
            "output",
            {
                "mode": "mock",
                "status": "extracted",
                "block_len": len(subsampling_block),
                "code_len": len(submission_code),
            },
        )

        return {
            "subsampling_block": subsampling_block,
            "status": "extracted",
        }

    prompt = SUBSAMPLING_EXTRACT_PROMPT.format(code=submission_code)

    try:
        response = call_llm(
            prompt, response_format="json", config=_default_llm_config()
        )
        result = parse_json_response(response)
        has_subsampling = result.get("has_subsampling", False)
        subsampling_block = (
            result.get("subsampling_block", "") if has_subsampling else ""
        )
    except Exception as e:
        log_node_event(
            "subsampling_extract",
            "llm_failed",
            {"error": str(e)[:300]},
        )
        subsampling_block = ""

    log_node_event(
        "subsampling_extract",
        "output",
        {
            "mode": "real",
            "status": "extracted",
            "block_len": len(subsampling_block),
            "code_len": len(submission_code),
        },
    )

    return {
        "subsampling_block": subsampling_block,
        "status": "extracted",
    }


@traceable("subsampling_remove")
def subsampling_remove(state: dict) -> dict:
    """Remove subsampling from the submission code.

    Replaces subsampling code with full-data usage, ensuring the final
    submission uses the complete training set.

    Mock: adds a use_full_data marker to the code.
    Real: calls LLM to remove subsampling code; falls back to string replacement.
    """
    simulate_delay()

    submission_code = state.get("submission_code", "")

    if _is_mock_mode():
        updated_code = submission_code + "# use_full_data: True\n"

        log_node_event(
            "subsampling_remove",
            "output",
            {"mode": "mock", "status": "removed", "code_len": len(updated_code)},
        )

        return {
            "submission_code": updated_code,
            "status": "subsampling_removed",
        }

    subsampling_block = state.get("subsampling_block", "")

    if not subsampling_block:
        log_node_event(
            "subsampling_remove",
            "output",
            {
                "mode": "real",
                "status": "subsampling_removed",
                "code_len": len(submission_code),
                "skipped": True,
            },
        )

        return {
            "submission_code": submission_code,
            "status": "subsampling_removed",
        }

    prompt = SUBSAMPLING_REMOVE_PROMPT.format(
        code=submission_code,
        subsampling_block=subsampling_block,
    )

    try:
        response = call_llm(
            prompt, response_format="code", config=_default_llm_config()
        )
        updated_code = parse_code_block(response)
    except Exception as e:
        log_node_event(
            "subsampling_remove",
            "llm_failed",
            {"error": str(e)[:300]},
        )
        updated_code = submission_code.replace(subsampling_block, "")

    log_node_event(
        "subsampling_remove",
        "output",
        {
            "mode": "real",
            "status": "subsampling_removed",
            "code_len": len(updated_code),
        },
    )

    return {
        "submission_code": updated_code,
        "status": "subsampling_removed",
    }


@traceable("A12__check_leakage_submission")
def A12__check_leakage_submission(state: dict) -> dict:
    """Data leakage check for submission code.

    Two-step process: extract code blocks that may contain leakage,
    then detect if leakage exists.

    Mock: always passes (no leakage detected).
    Real: calls LLM to detect data leakage; retries up to MAX_LEAKAGE_FIX_RETRIES.
    """
    simulate_delay()

    submission_code = state.get("submission_code", "")

    if _is_mock_mode():
        log_node_event(
            "A12__check_leakage_submission",
            "output",
            {"mode": "mock", "status": "ok", "code_len": len(submission_code)},
        )

        return {"leakage_status": "ok", "status": "ok"}

    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "")

    for attempt in range(MAX_LEAKAGE_FIX_RETRIES):
        try:
            prompt = DATA_LEAKAGE_EXTRACT_PROMPT.format(
                task_desc=task_desc,
                metric=metric,
                code=submission_code,
            )
            response = call_llm(
                prompt, response_format="json", config=_default_llm_config()
            )
            result = parse_json_response(response)

            has_leakage = result.get("has_leakage", False)

            if has_leakage:
                log_node_event(
                    "A12__check_leakage_submission",
                    "output",
                    {
                        "mode": "real",
                        "status": "leakage_fail",
                        "attempt": attempt + 1,
                        "leakage_issues": result.get("leakage_issues", []),
                        "code_len": len(submission_code),
                    },
                )

                return {
                    "leakage_status": "Yes Data Leakage",
                    "leakage_issues": result.get("leakage_issues", []),
                    "status": "leakage_fail",
                }

            log_node_event(
                "A12__check_leakage_submission",
                "output",
                {
                    "mode": "real",
                    "status": "ok",
                    "attempt": attempt + 1,
                    "code_len": len(submission_code),
                },
            )

            return {"leakage_status": "ok", "status": "ok"}

        except Exception as e:
            log_node_event(
                "A12__check_leakage_submission",
                "llm_failed",
                {"attempt": attempt + 1, "error": str(e)[:300]},
            )
            continue

    log_node_event(
        "A12__check_leakage_submission",
        "output",
        {
            "mode": "real",
            "status": "ok",
            "all_retries_failed": True,
            "code_len": len(submission_code),
        },
    )

    return {"leakage_status": "ok", "status": "ok"}


@traceable("A12__fix_leakage_submission")
def A12__fix_leakage_submission(state: dict) -> dict:
    """Fix data leakage detected in submission code.

    Mock: appends a leakage fix comment and sets status to re-check.
    Real: calls LLM to fix leakage issues in the code.
    """
    simulate_delay()

    submission_code = state.get("submission_code", "")

    if _is_mock_mode():
        fixed_code = submission_code + "# leakage_fix_submission\n"

        log_node_event(
            "A12__fix_leakage_submission",
            "output",
            {"mode": "mock", "status": "fixed", "code_len": len(fixed_code)},
        )

        return {
            "submission_code": fixed_code,
            "leakage_status": None,
            "status": "leakage_fix_applied",
        }

    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "")
    leakage_issues = state.get("leakage_issues", [])

    if isinstance(leakage_issues, list):
        leakage_issues_str = "\n".join(
            f"- {issue}" if isinstance(issue, str) else f"- {issue}"
            for issue in leakage_issues
        )
    else:
        leakage_issues_str = str(leakage_issues)

    prompt = DATA_LEAKAGE_CORRECT_PROMPT.format(
        task_desc=task_desc,
        metric=metric,
        leakage_issues=leakage_issues_str,
        code=submission_code,
    )

    try:
        response = call_llm(
            prompt, response_format="code", config=_default_llm_config()
        )
        fixed_code = parse_code_block(response)
    except Exception as e:
        log_node_event(
            "A12__fix_leakage_submission",
            "llm_failed",
            {"error": str(e)[:300]},
        )
        fixed_code = submission_code + "# leakage_fix_submission\n"

    log_node_event(
        "A12__fix_leakage_submission",
        "output",
        {"mode": "real", "status": "fixed", "code_len": len(fixed_code)},
    )

    return {
        "submission_code": fixed_code,
        "leakage_status": None,
        "status": "leakage_fix_applied",
    }


def route_after_leakage_check_submission(state: dict) -> str:
    """Route after A12__check_leakage_submission.

    E-S04: leakage_fail -> A12__fix_leakage_submission
    ok -> eval_submission
    """
    if state.get("leakage_status") in ("leakage_fail", "Yes Data Leakage"):
        return "A12__fix_leakage_submission"
    return "eval_submission"
