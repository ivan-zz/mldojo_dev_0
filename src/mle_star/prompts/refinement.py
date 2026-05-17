"""Prompt templates for refinement phase: CODER_PROMPT, PLANNER_PROMPT.

Used by nodes/refinement.py (A7, A8) to implement refined code blocks
and generate refinement plans.
"""

CODER_PROMPT = """You are an expert ML engineer refining a specific code block in a machine learning pipeline.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric} ({direction})

CURRENT FULL SOLUTION:
```python
{current_solution}
```

TARGET BLOCK TO REFINE:
```python
{target_block}
```

IMPROVEMENT PLAN:
{current_plan}

PREVIOUS ATTEMPTS (if any):
{previous_attempts}

INSTRUCTIONS:
1. Rewrite ONLY the target block to improve the overall pipeline performance.
2. The refined block must be a DROP-IN REPLACEMENT for the target block — same function name, same inputs/outputs.
3. Focus on the improvements described in the plan.
4. The block should work correctly when substituted into the full solution.
5. For multi-target prediction, handle each target appropriately.
6. Use only sklearn and standard libraries (numpy, pandas, scipy).

Return ONLY the refined code block (not the full solution) in a Python code block:
```python
# refined version of the target block
```"""

PLANNER_PROMPT = """You are an expert ML engineer planning the next refinement step for a machine learning pipeline.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

TARGET BLOCK:
```python
{target_block}
```

CURRENT SOLUTION PERFORMANCE: {best_score}

PREVIOUS PLANS AND SCORES:
{plan_history}

EXECUTION OUTPUT FROM LAST ATTEMPT:
{execution_output}

INSTRUCTIONS:
1. Analyze the previous attempts and their scores.
2. If the last attempt improved the score, try a different approach to push further.
3. If the last attempt did not improve, try an alternative strategy.
4. Be specific about what changes to make to the target block.
5. Do NOT repeat strategies that have already been tried.

Provide a detailed, specific improvement plan for the next iteration."""
