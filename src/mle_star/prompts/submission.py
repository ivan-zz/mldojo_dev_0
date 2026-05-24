"""Prompt templates for submission phase: SUBMISSION_PROMPT, SUBSAMPLING_EXTRACT_PROMPT, SUBSAMPLING_REMOVE_PROMPT.

Used by nodes/submission.py (A_test__submit, subsampling_extract, subsampling_remove).
"""

SUBMISSION_PROMPT = """You are preparing a final ML submission script. Clean up the following solution for Kaggle-style submission.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

BEST SOLUTION:
```python
{solution}
```

BEST VALIDATION SCORE: {best_score}

INSTRUCTIONS:
1. Remove any debugging code, print statements (except the final score), and experimental code.
2. Ensure the script reads train.csv and test.csv from the current directory.
3. Ensure the script generates a submission.csv file with the correct format (id + predictions).
4. For multi-target prediction, include ALL target columns in the submission.
5. Keep ALL the best-performing model logic intact — do not simplify or remove features.
6. Make sure the final score is printed as: Final Validation Performance: <score>
7. Handle edge cases: missing values, unseen categories, etc.

Return ONLY the clean submission script. Do NOT wrap it in markdown code fences. Start your response directly with "import" or "from"."""

SUBSAMPLING_EXTRACT_PROMPT = """You are analyzing an ML script to identify any data subsampling that was used for faster iteration during development.

CODE TO ANALYZE:
```python
{code}
```

INSTRUCTIONS:
Look for subsampling patterns such as:
- `df.sample(n=...)` or `df.sample(frac=...)`
- `train_test_split(..., train_size=...)` with train_size < 1.0 without CV
- Slicing like `train[:10000]` or `train.iloc[:N]` for training data only
- Any explicit train size reduction that doesn't use cross-validation
- Hardcoded row limits like `n_samples = 10000`

Do NOT flag:
- Cross-validation splits (KFold, StratifiedKFold) — these are proper evaluation
- Test set handling — that's required
- Feature selection — that's a legitimate technique

Respond in JSON format:
```json
{{
  "has_subsampling": true/false,
  "subsampling_block": "the exact code lines that perform subsampling, or empty string if none"
}}
```"""

SUBSAMPLING_REMOVE_PROMPT = """You are removing data subsampling from an ML script to use the full training dataset for the final submission.

CODE WITH SUBSAMPLING:
```python
{code}
```

SUBSAMPLING BLOCK IDENTIFIED:
{subsampling_block}

INSTRUCTIONS:
1. Remove or replace the subsampling code so the full training dataset is used.
2. Common fixes:
   - Replace `df.sample(n=10000)` with using the full dataframe
   - Replace `train.iloc[:N]` with `train` (full dataset)
   - Replace `train_size=0.1` in train_test_split with cross-validation
   - Remove hardcoded sample size variables
3. Keep ALL other logic intact — feature engineering, model training, etc.
4. The script must still produce Final Validation Performance: <score>
5. For multi-target prediction, keep handling for all targets.

Return ONLY the modified Python script. Do NOT wrap it in markdown code fences. Start your response directly with "import" or "from"."""
