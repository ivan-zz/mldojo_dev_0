"""Ablation phase nodes: A4__generate_ablation, A5__summarize_ablation, A6__extract_block.

Mock implementations for Stage 3. Real implementations built in Stage 8.

Key design change: A4 now generates SEPARATE scripts (baseline + 2-3 ablation
variants) instead of a single monolithic script. Each variant is executed
independently via the ablation_variant_subgraph with debug retry support.
"""

import os

from src.mle_star.state.shared import (
    traceable,
    simulate_delay,
    log_node_event,
    random_score,
    call_llm,
    _default_llm_config,
    parse_json_response,
    parse_code_block,
    parse_code_blocks,
    fuzzy_replace,
    format_direction,
)
from src.mle_star.config import MOCK_MODE
from src.mle_star.prompts.ablation import (
    ABLATION_STUDY_PROMPT,
    ABLATION_SUMMARIZE_PROMPT,
    EXTRACTOR_PROMPT,
)


def _is_mock_mode():
    return MOCK_MODE or os.environ.get("MLE_MOCK_MODE", "").lower() in (
        "1",
        "true",
        "yes",
    )


MOCK_SOLUTION_BLOCKS = [
    {
        "name": "preprocess",
        "code": "def preprocess(df):\n    df = df.dropna()\n    df = df.fillna(0)\n    return df\n",
        "start_line": 1,
        "end_line": 4,
        "type": "FunctionDef",
    },
    {
        "name": "feature_engineering",
        "code": "def feature_engineering(df):\n    df['new_feat'] = df['a'] * df['b']\n    return df\n",
        "start_line": 6,
        "end_line": 8,
        "type": "FunctionDef",
    },
    {
        "name": "train_model",
        "code": "def train_model(X, y):\n    from sklearn.ensemble import RandomForestClassifier\n    model = RandomForestClassifier(n_estimators=100)\n    model.fit(X, y)\n    return model\n",
        "start_line": 10,
        "end_line": 14,
        "type": "FunctionDef",
    },
]


@traceable("A4__generate_ablation")
def A4__generate_ablation(state: dict) -> dict:
    """Generate separate ablation scripts for baseline + variant execution.

    Parses the current solution into functional blocks using AST (mock: uses
    predefined blocks). Returns ablation_scripts: a list of dicts, each with
    {name, code, block_name}. The first entry is always the baseline (full
    solution), followed by 2-3 ablation variants (each disabling one component).

    A4 from MLE-STAR paper.
    """
    simulate_delay()

    if _is_mock_mode():
        return _A4__generate_ablation_mock(state)

    solution = state.get("current_solution", "")
    blocks = []
    if solution and "def " in solution:
        blocks = parse_code_blocks(solution)
    if not blocks:
        blocks = MOCK_SOLUTION_BLOCKS

    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "accuracy")
    metric_direction = state.get("metric_direction", "maximize")
    previous_summaries = state.get("previous_summaries", [])
    if isinstance(previous_summaries, list):
        previous_summaries_str = "\n".join(str(s) for s in previous_summaries)
    else:
        previous_summaries_str = str(previous_summaries)

    func_blocks_str = "\n".join(
        f"- {b['name']} ({b['type']}, lines {b['start_line']}-{b['end_line']}): {b['code'][:200]}"
        for b in blocks
    )

    try:
        prompt = ABLATION_STUDY_PROMPT.format(
            task_desc=task_desc,
            metric=metric,
            direction=format_direction(metric_direction),
            solution=solution,
            functional_blocks=func_blocks_str,
            previous_summaries=previous_summaries_str,
        )
        response = call_llm(prompt, response_format="json")
        parsed = parse_json_response(response)
        ablation_scripts = parsed.get("ablation_scripts", [])
        if not ablation_scripts:
            raise ValueError("No ablation_scripts in LLM response")
        for script in ablation_scripts:
            script.setdefault("name", script.get("block_name", "unknown"))
            script.setdefault("code", "")
            script.setdefault("block_name", script.get("name", "unknown"))
    except Exception as e:
        log_node_event(
            "A4__generate_ablation",
            "llm_fallback_to_mock",
            {"error": str(e)[:300]},
        )
        return _A4__generate_ablation_mock(state)

    log_node_event(
        "A4__generate_ablation",
        "output",
        {
            "blocks_found": len(blocks),
            "scripts_generated": len(ablation_scripts),
            "mode": "real",
        },
    )

    return {
        "ablation_scripts": ablation_scripts,
        "functional_blocks": blocks,
        "status": "ablation_generated",
    }


def _A4__generate_ablation_mock(state: dict) -> dict:
    """Mock implementation for A4__generate_ablation."""
    blocks = []
    solution = state.get("current_solution", "")
    if solution and "def " in solution:
        blocks = _parse_code_blocks_mock(solution)
    if not blocks:
        blocks = MOCK_SOLUTION_BLOCKS

    ablation_scripts = _generate_ablation_scripts_mock(blocks, solution)

    log_node_event(
        "A4__generate_ablation",
        "output",
        {
            "blocks_found": len(blocks),
            "scripts_generated": len(ablation_scripts),
            "mode": "mock",
        },
    )

    return {
        "ablation_scripts": ablation_scripts,
        "functional_blocks": blocks,
        "status": "ablation_generated",
    }


@traceable("A5__summarize_ablation")
def A5__summarize_ablation(state: dict) -> dict:
    """Summarize ablation results into a human-readable summary.

    Receives ablation_results_list aggregated from all variant executions
    (baseline + ablation variants). Creates a summary identifying the
    highest-impact component by comparing each variant's score to baseline.

    Mock: returns a summary identifying the highest-impact component.
    In real implementation (Stage 8), uses LLM to parse raw execution
    output into a structured summary.
    """
    simulate_delay()

    if _is_mock_mode():
        return _A5__summarize_ablation_mock(state)

    results_list = state.get("ablation_results_list", [])
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "accuracy")

    ablation_results_str = "\n".join(
        f"- Variant: {r.get('block_name', 'unknown')}, Score: {r.get('score', 'N/A')}, Impact: {r.get('impact', 'N/A')}"
        for r in results_list
    )

    try:
        prompt = ABLATION_SUMMARIZE_PROMPT.format(
            task_desc=task_desc,
            metric=metric,
            ablation_results=ablation_results_str,
        )
        summary = call_llm(prompt)
        if not summary or not summary.strip():
            raise ValueError("Empty LLM response for ablation summary")
    except Exception as e:
        log_node_event(
            "A5__summarize_ablation",
            "llm_fallback_to_mock",
            {"error": str(e)[:300]},
        )
        return _A5__summarize_ablation_mock(state)

    log_node_event(
        "A5__summarize_ablation",
        "output",
        {"summary_len": len(summary), "mode": "real"},
    )

    return {
        "ablation_summaries": [summary],
    }


def _A5__summarize_ablation_mock(state: dict) -> dict:
    """Mock implementation for A5__summarize_ablation."""
    results_list = state.get("ablation_results_list", [])

    if results_list:
        ablation_variants = [
            r for r in results_list if r.get("block_name") != "baseline"
        ]
        if ablation_variants:
            sorted_variants = sorted(
                ablation_variants,
                key=lambda x: x.get("impact", 0),
                reverse=True,
            )
            top = sorted_variants[0]
            summary = (
                f"Ablation summary: {top.get('block_name', 'unknown')} has the largest impact "
                f"(impact={top.get('impact', 0):.4f}). "
            )
            if len(sorted_variants) > 1:
                mid = sorted_variants[1]
                summary += f"{mid.get('block_name', 'unknown')} has moderate impact ({mid.get('impact', 0):.4f}). "
            if len(sorted_variants) > 2:
                for r in sorted_variants[2:]:
                    summary += f"{r.get('block_name', 'unknown')} has negligible impact ({r.get('impact', 0):.4f}). "
        else:
            summary = "Ablation summary: Only baseline results available, no variants executed."
    else:
        summary = "Ablation summary: No results available."

    log_node_event(
        "A5__summarize_ablation",
        "output",
        {"summary_len": len(summary), "mode": "mock"},
    )

    return {
        "ablation_summaries": [summary],
    }


@traceable("A6__extract_block")
def A6__extract_block(state: dict) -> dict:
    """Extract the target code block with the highest impact score and generate initial plan.

    Uses ablation_results_list to identify the single most impactful component,
    then extracts the original code block for that component and drafts
    the first refinement plan (p_0).

    Respects previous_blocks to avoid re-refining the same component.
    The extractor is given a list of all code blocks that have already been
    refined in previous outer loops and must skip them.

    Mock: returns the block with highest impact (excluding previously refined)
    and a mock plan.
    """
    simulate_delay()

    if _is_mock_mode():
        return _A6__extract_block_mock(state)

    results_list = state.get("ablation_results_list", [])
    blocks = state.get("functional_blocks", [])
    if not blocks:
        blocks = MOCK_SOLUTION_BLOCKS

    solution = state.get("current_solution", "")
    task_desc = state.get("task_desc", "")
    metric = state.get("metric", "accuracy")
    previous_blocks = state.get("previous_blocks", [])
    previous_summaries = state.get("previous_summaries", [])

    ablation_summary = ""
    ablation_summaries = state.get("ablation_summaries", [])
    if ablation_summaries:
        ablation_summary = "\n".join(str(s) for s in ablation_summaries)

    func_blocks_str = "\n".join(
        f"- {b['name']} ({b['type']}): {b['code'][:300]}" for b in blocks
    )
    previous_blocks_str = ", ".join(previous_blocks) if previous_blocks else "None"

    try:
        prompt = EXTRACTOR_PROMPT.format(
            task_desc=task_desc,
            metric=metric,
            ablation_summary=ablation_summary,
            solution=solution,
            functional_blocks=func_blocks_str,
            previous_blocks=previous_blocks_str,
        )
        response = call_llm(prompt, response_format="json")
        parsed = parse_json_response(response)

        target_name = parsed.get("target_block_name", "")
        target_code = parsed.get("target_block_code", "")
        initial_plan = parsed.get("initial_plan", "")

        if not target_name:
            raise ValueError("No target_block_name in LLM response")

        resolved_code = target_code
        if not resolved_code or resolved_code not in solution:
            for b in blocks:
                if b.get("name") == target_name:
                    resolved_code = b.get("code", "")
                    break
            if not resolved_code or resolved_code not in solution:
                resolved_code = target_code

    except Exception as e:
        log_node_event(
            "A6__extract_block",
            "llm_fallback_to_mock",
            {"error": str(e)[:300]},
        )
        return _A6__extract_block_mock(state)

    log_node_event(
        "A6__extract_block",
        "output",
        {
            "target_block_name": target_name,
            "plan_len": len(initial_plan),
            "mode": "real",
        },
    )

    return {
        "target_block": resolved_code,
        "initial_plan": initial_plan,
        "status": "block_extracted",
    }


def _A6__extract_block_mock(state: dict) -> dict:
    """Mock implementation for A6__extract_block."""
    results_list = state.get("ablation_results_list", [])
    blocks = state.get("functional_blocks", [])
    if not blocks:
        blocks = MOCK_SOLUTION_BLOCKS

    previous_blocks = state.get("previous_blocks", [])

    ablation_variants = [r for r in results_list if r.get("block_name") != "baseline"]

    available_variants = [
        r for r in ablation_variants if r.get("block_name", "") not in previous_blocks
    ]

    if not available_variants:
        available_variants = ablation_variants

    if available_variants:
        sorted_results = sorted(
            available_variants,
            key=lambda x: x.get("impact", 0),
            reverse=True,
        )
        target_name = sorted_results[0].get("block_name", "unknown")
    elif ablation_variants:
        sorted_results = sorted(
            ablation_variants,
            key=lambda x: x.get("impact", 0),
            reverse=True,
        )
        target_name = sorted_results[0].get("block_name", "unknown")
    else:
        target_name = blocks[0].get("name", "unknown") if blocks else "unknown"

    target_block = ""
    for b in blocks:
        if b.get("name") == target_name:
            target_block = b.get("code", "")
            break
    if not target_block:
        target_block = blocks[0].get("code", "") if blocks else "# no block found"

    initial_plan = (
        f"Improve {target_name} by enhancing feature preprocessing "
        f"and adding regularization to prevent overfitting."
    )

    log_node_event(
        "A6__extract_block",
        "output",
        {
            "target_block_name": target_name,
            "plan_len": len(initial_plan),
            "mode": "mock",
        },
    )

    return {
        "target_block": target_block,
        "initial_plan": initial_plan,
        "status": "block_extracted",
    }


def _parse_code_blocks_mock(solution: str) -> list[dict]:
    """Mock code block parser. Real version uses AST in Stage 8."""
    if not solution:
        return []
    blocks = []
    lines = solution.splitlines()
    current_name = None
    current_start = None
    current_lines = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("def "):
            if current_name:
                blocks.append(
                    {
                        "name": current_name,
                        "code": "\n".join(current_lines),
                        "start_line": current_start,
                        "end_line": i - 1,
                        "type": "FunctionDef",
                    }
                )
            current_name = stripped.split("(")[0].replace("def ", "")
            current_start = i
            current_lines = [line]
        elif current_name:
            current_lines.append(line)
    if current_name:
        blocks.append(
            {
                "name": current_name,
                "code": "\n".join(current_lines),
                "start_line": current_start,
                "end_line": len(lines),
                "type": "FunctionDef",
            }
        )
    return blocks


def _generate_ablation_scripts_mock(blocks: list[dict], solution: str) -> list[dict]:
    """Generate separate scripts for baseline + ablation variants.

    Returns a list of dicts, each with {name, code, block_name}:
    - First entry: baseline (full solution)
    - Remaining entries: ablation variants (each disabling one component)
    """
    scripts = [
        {
            "name": "baseline",
            "code": solution or "# baseline solution",
            "block_name": "baseline",
        }
    ]

    for b in blocks[1:]:
        name = b.get("name", "unknown")
        code = _generate_variant_code_mock(solution, b)
        scripts.append(
            {
                "name": f"ablation_{name}",
                "code": code,
                "block_name": name,
            }
        )

    return scripts


def _generate_ablation_scripts_real(
    ablation_scripts_data: list[dict],
) -> list[dict]:
    """Use LLM-generated ablation scripts from A4's response.

    Takes the parsed JSON response from A4 and normalizes the entries
    to ensure each has name, code, and block_name keys.
    """
    scripts = []
    for entry in ablation_scripts_data:
        scripts.append(
            {
                "name": entry.get("name", entry.get("block_name", "unknown")),
                "code": entry.get("code", ""),
                "block_name": entry.get("block_name", entry.get("name", "unknown")),
            }
        )
    return scripts


def _generate_variant_code_mock(solution: str, block: dict) -> str:
    """Generate mock ablation variant code that disables one component.

    Mock: returns a modified version of the solution with the target
    block commented out. Real implementation uses the LLM (A4 prompt).
    """
    block_name = block.get("name", "unknown")
    block_code = block.get("code", "")

    if solution and block_code and block_code in solution:
        disabled = f"# [ABLATION] {block_name} disabled\n"
        variant = solution.replace(block_code, disabled + block_code)
        variant = variant + f"\n# Ablation: {block_name} disabled"
    else:
        variant = f"# Mock ablation variant for {block_name}\n{solution or ''}"

    return variant
