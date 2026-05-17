"""Entry point for MLE-STAR full pipeline.

Usage:
    python main.py [options]

Options:
    --task-desc PATH          Path to task description file (default: input/description.md)
    --score-function-desc STR Score function description (default: inferred from task desc)
    --phase PHASE             Start from specific phase: search, ensemble, submission
    --max-full-cycles N       Override max_full_cycles (default: 3)
    --num-parallel-solutions N Override L parallel pipelines (default: 2)
    --provider PROVIDER       LLM provider: ollama, openrouter, openai (default: from env)
    --model MODEL             LLM model name (default: from env)
    --mock                    Enable mock mode (no real LLM calls or subprocess execution)
    --resume RUN_DIR          Resume from checkpoint directory
    --run-dir RUN_DIR         Specify run directory
    --fast                    Shortcut: max_full_cycles=1, max_outer_steps=1,
                              max_inner_steps=1, max_ensemble_rounds=1,
                              num_parallel_solutions=1

    Algorithm 1 only:
        from src.mle_star.algorithms.algorithm_1 import run as run_alg1
        result = run_alg1(run_dir="./runs/my_run")

    Full pipeline:
        from src.mle_star.graph import run
        result = run(run_dir="./runs/my_run")
"""

import argparse
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mle_star.graph import run
from src.mle_star.supervisor import SupervisorConfig
from src.mle_star.state.shared import infer_metric_direction

DEFAULT_TASK_DESC = "ML pipeline optimization"
DEFAULT_SCORE_FUNCTION_DESC = ""


def _detect_datasets():
    """Auto-detect dataset files in the input/ directory."""
    datasets = []
    input_dir = Path(__file__).parent / "input"
    if input_dir.is_dir():
        train_csv = input_dir / "train.csv"
        test_csv = input_dir / "test.csv"
        if train_csv.is_file():
            datasets.append(str(train_csv))
        if test_csv.is_file():
            datasets.append(str(test_csv))
    return datasets


def _load_task_desc(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


def _build_initial_state(args) -> dict:
    task_desc = DEFAULT_TASK_DESC
    score_function_desc = DEFAULT_SCORE_FUNCTION_DESC

    if args.task_desc:
        desc_path = Path(args.task_desc)
        if desc_path.is_file():
            task_desc = _load_task_desc(str(desc_path))
        else:
            task_desc = args.task_desc

    if args.score_function_desc:
        score_function_desc = args.score_function_desc
    elif task_desc != DEFAULT_TASK_DESC:
        section = task_desc.lower()
        if "rmsle" in section or "rmse" in section or "mae" in section:
            score_function_desc = "RMSLE"
        elif "accuracy" in section or "auc" in section:
            score_function_desc = "accuracy"
        else:
            score_function_desc = ""

    metric_direction = (
        infer_metric_direction(score_function_desc)
        if score_function_desc
        else "maximize"
    )

    state = {
        "task_desc": task_desc,
        "datasets": _detect_datasets(),
        "score_function_desc": score_function_desc,
        "metric_direction": metric_direction,
        "phase": "search",
        "phase_history": [],
        "current_solution": "",
        "best_solution": "",
        "best_score": 0.0,
        "current_score": None,
        "raw_best_score": None,
        "alg1_result": {},
        "parallel_results": [],
        "alg2_result": {},
        "outer_step": 0,
        "inner_step": 0,
        "convergence_achieved": False,
        "alg3_result": {},
        "ensemble_solutions": [],
        "ensemble_input_scores": [],
        "ensemble_round": 0,
        "submission_code": "",
        "submission_score": None,
        "full_cycles": 0,
        "debug_history": [],
        "security_violations": [],
        "stage_history": [],
        "status": "start",
    }

    if args.phase == "ensemble":
        state["phase"] = "search_done"
        state["best_solution"] = (
            state.get("best_solution", "") or "mock_solution_for_ensemble"
        )
        state["best_score"] = state.get("best_score", 0) or 0.85
    elif args.phase == "submission":
        state["phase"] = "submission"
        state["best_solution"] = state.get("best_solution", "") or "mock_final_solution"
        state["best_score"] = state.get("best_score", 0) or 0.90

    state["max_full_cycles"] = args.max_full_cycles

    return state


def _build_config(args) -> SupervisorConfig:
    from src.mle_star.state.shared import LLMConfig

    config = SupervisorConfig()
    config.max_full_cycles = args.max_full_cycles
    config.num_parallel_solutions = args.num_parallel_solutions

    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    if provider:
        llm_config = LLMConfig(
            provider=provider,
            model=model or "",
        )
        config.llm_config = llm_config

    if getattr(args, "fast", False):
        config.max_full_cycles = 1
        config.max_outer_steps = 1
        config.max_inner_steps = 1
        config.max_ensemble_rounds = 1
        config.num_parallel_solutions = 1

    if getattr(args, "mock", False):
        os.environ["MLE_MOCK_MODE"] = "1"

    return config


def main():
    parser = argparse.ArgumentParser(description="MLE-STAR full pipeline")
    parser.add_argument(
        "--task-desc",
        default="input/description.md",
        help="Path to task description file (default: input/description.md)",
    )
    parser.add_argument(
        "--score-function-desc",
        default="",
        help="Score function description (default: inferred from task desc)",
    )
    parser.add_argument(
        "--phase",
        default="search",
        choices=["search", "ensemble", "submission"],
        help="Start from specific phase (default: search)",
    )
    parser.add_argument(
        "--max-full-cycles",
        type=int,
        default=3,
        help="Maximum full pipeline cycles before forced submission (default: 3)",
    )
    parser.add_argument(
        "--num-parallel-solutions",
        type=int,
        default=2,
        help="Number of parallel pipeline solutions L (default: 2)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Specify run directory",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut: max_full_cycles=1, T=1, K=1, R=1, L=1 for quick testing",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Enable mock mode (no real LLM calls or subprocess execution)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["ollama", "openrouter", "openai"],
        help="LLM provider (default: from env LLM_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: from env LLM_MODEL or OLLAMA_MODEL)",
    )

    args = parser.parse_args()

    config = _build_config(args)

    if args.resume:
        initial_state = None
    else:
        initial_state = _build_initial_state(args)
        initial_state["max_full_cycles"] = config.max_full_cycles
        initial_state.setdefault(
            "num_parallel_solutions", config.num_parallel_solutions
        )

    start_time = time.time()
    result = run(
        initial_state=initial_state,
        run_dir=args.run_dir if not args.resume else args.resume,
        config=config,
        resume=bool(args.resume),
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  MLE-STAR Pipeline Summary")
    print("=" * 60)
    print(f"  Status:           {result.get('status')}")
    print(f"  Phase:            {result.get('phase')}")
    print(f"  Full cycles:      {result.get('full_cycles')}")
    print(f"  Metric direction: {result.get('metric_direction', 'unknown')}")
    print(f"  Best score:       {result.get('best_score')}")
    print(f"  Best solution:    {result.get('best_solution', '')[:80]}...")
    print(f"  Submission:       {result.get('submission_code', '')[:80]}...")
    print(f"  Total time:       {elapsed:.1f}s")

    phase_history = result.get("phase_history", [])
    if phase_history:
        print(f"\n  Phase transitions ({len(phase_history)} entries):")
        for ph in phase_history:
            print(f"    {ph.get('phase', '?'):15s} decision={ph.get('decision', '?')}")

    stage_history = result.get("stage_history", [])
    if stage_history:
        print(f"\n  Stage history ({len(stage_history)} entries, first 15 shown):")
        for sh in stage_history[:15]:
            print(f"    {sh.get('stage', '?'):30s} status={sh.get('status', '?')}")
        if len(stage_history) > 15:
            print(f"    ... and {len(stage_history) - 15} more entries")

    print("=" * 60)


if __name__ == "__main__":
    main()
