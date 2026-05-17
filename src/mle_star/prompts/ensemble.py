"""Prompt templates for ensemble phase: ENSEMBLE_PLANNER_PROMPT, ENSEMBLER_PROMPT.

Used by nodes/ensemble.py (A9, A10) to plan and implement ensemble strategies.
"""

ENSEMBLE_PLANNER_PROMPT = """You are an expert ML ensembling strategist. Plan an ensemble strategy to combine multiple ML pipeline solutions.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric} ({direction})

SOLUTION SCORES:
{score_descriptions}

PREVIOUS ENSEMBLE PLANS (do NOT repeat these strategies):
{previous_plans}

CURRENT BEST ENSEMBLE SCORE: {best_ensemble_score}

ENSEMBLE ROUND: {ensemble_round}

INSTRUCTIONS:
1. Analyze the strengths and weaknesses of each solution based on their scores.
2. Propose an ensemble strategy (e.g., weighted average, stacking, blending, voting).
3. The strategy should complement the individual solutions' strengths.
4. Consider: simple average, weighted average (weight by inverse score), stacking with a meta-learner, feature concatenation + ensemble.
5. Be specific about how to combine the predictions.

Provide a clear, structured plan describing:
- Which solutions to include
- The ensemble method (weighted average, stacking, etc.)
- Specific weights or strategies
- Why this combination should improve performance"""

ENSEMBLER_PROMPT = """You are an expert ML engineer. Implement the following ensemble strategy as a complete Python script.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric} ({direction})

ENSEMBLE PLAN:
{plan}

SOLUTIONS TO ENSEMBLE:
{solutions_with_scores}

{best_ensemble_section}

INSTRUCTIONS:
1. Create a COMPLETE, self-contained Python script that implements the ensemble strategy.
2. The script must:
   - Import only sklearn and standard libraries (numpy, pandas, scipy)
   - Read train.csv and test.csv from the current directory
   - Implement the ensemble strategy described in the plan
   - Use 5-fold cross-validation
   - Print: Final Validation Performance: <score>
3. For multi-target prediction, handle each target appropriately.
4. For weighted averaging, weight models by their inverse validation scores (better models get more weight).
5. Ensure all data handling (missing values, feature alignment) is consistent across models.

Return ONLY the complete Python script in a code block:
```python
# ensemble implementation
```"""
