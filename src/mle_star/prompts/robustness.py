"""Prompt templates for robustness: DATA_USAGE_CHECK_PROMPT, DATA_USAGE_FIX_PROMPT,
DATA_LEAKAGE_EXTRACT_PROMPT, DATA_LEAKAGE_CORRECT_PROMPT, DEBUG_PROMPT,
ERROR_CLASSIFIER_PROMPT.

These prompts are used by the robustness nodes (A13__check_usage, A13__fix_usage,
A12__check_leakage, A12__fix_leakage, A11__debug) during the search phase.
"""

DATA_USAGE_CHECK_PROMPT = """You are an expert ML code reviewer. Check whether the following ML pipeline code uses only allowed data sources.

TASK DESCRIPTION:
{task_desc}

ALLOWED DATA SOURCES:
- train.csv (training data)
- test.csv (test data, for prediction only — NOT for training)
- Any derived features from train.csv

CODE TO CHECK:
```python
{code}
```

Check for these violations:
1. Using external data files beyond train.csv and test.csv
2. Using pre-trained models or external knowledge not derived from train.csv
3. Using test.csv target values for training (this is also a leakage issue)
4. Using data from sources other than the provided CSV files

Respond in JSON format:
```json
{{
  "usage_ok": true/false,
  "violations": ["description of each violation"],
  "explanation": "Brief explanation of the decision"
}}
```"""

DATA_USAGE_FIX_PROMPT = """You are an expert ML code reviewer. Fix the data usage violations in the following ML pipeline code.

TASK DESCRIPTION:
{task_desc}

VIOLATIONS FOUND:
{violations}

ORIGINAL CODE:
```python
{code}
```

Fix the code to remove all data usage violations. Only use train.csv for training and test.csv for prediction (no target values). Remove any references to external data sources.

Return ONLY the fixed code in a Python code block:
```python
# fixed code here
```

The fixed code must still be a complete, runnable Python script that prints:
Final Validation Performance: <score>"""

DATA_LEAKAGE_EXTRACT_PROMPT = """You are an expert ML code reviewer. Extract and identify potential data leakage in the following ML pipeline code.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

CODE TO ANALYZE:
```python
{code}
```

Data leakage occurs when information from outside the training dataset is used to create the model, giving the illusion of better performance. Common types:
1. Target leakage: Using target/label information during feature engineering (e.g., mean-encoding with target)
2. Train-test contamination: Fitting preprocessors (scalers, imputers, encoders) on the entire dataset before splitting
3. Look-ahead bias: Using future data that wouldn't be available at prediction time
4. K-fold leakage: Applying transformations before cross-validation split

For each leakage issue found:
- Identify the exact code lines
- Explain why it's leakage
- Classify the type (target_leakage, train_test_contamination, look_ahead, kfold_leakage)

Respond in JSON format:
```json
{{
  "has_leakage": true/false,
  "leakage_issues": [
    {{
      "type": "leakage_type",
      "location": "line numbers or code section",
      "description": "why this is leakage",
      "severity": "critical/moderate/minor"
    }}
  ],
  "explanation": "Brief summary"
}}
```"""

DATA_LEAKAGE_CORRECT_PROMPT = """You are an expert ML code fixer. Fix the data leakage issues in the following ML pipeline code.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

LEAKAGE ISSUES FOUND:
{leakage_issues}

ORIGINAL CODE:
```python
{code}
```

Fix ALL leakage issues. Common fixes:
1. Target leakage: Remove features derived from target, or compute them only within CV folds
2. Train-test contamination: Fit preprocessors only on training fold data inside cross-validation
3. K-fold leakage: Use Pipeline with cross-validation to ensure proper data separation

IMPORTANT: The fixed code must be a COMPLETE, runnable Python script that:
1. Uses only sklearn and standard libraries
2. Reads train.csv and test.csv from current directory
3. Uses proper 5-fold cross-validation with the {metric} metric
4. Prints: Final Validation Performance: <score>

Return ONLY the fixed code:
```python
# fixed code here
```"""

DEBUG_PROMPT = """You are an expert Python/ML debugger. The following ML pipeline code crashed during execution. Fix the error.

TASK DESCRIPTION:
{task_desc}

CODE THAT CRASHED:
```python
{code}
```

ERROR MESSAGE:
```
{error_message}
```

STDERR OUTPUT:
```
{stderr}
```

Fix the code to resolve the error. Common issues:
1. Import errors: missing module imports or typos
2. Shape mismatches: incorrect array dimensions
3. NaN/inf values: handle missing data properly
4. Indexing errors: incorrect column references
5. Type errors: incorrect data types

IMPORTANT: The fixed code must be a COMPLETE, runnable Python script that:
1. Uses only sklearn and standard libraries
2. Reads train.csv and test.csv from current directory
3. Uses 5-fold cross-validation with the {metric} metric
4. Prints: Final Validation Performance: <score>

Return ONLY the fixed code:
```python
# fixed code here
```"""

ERROR_CLASSIFIER_PROMPT = """You are an expert Python/ML error classifier. Classify the following error into one of these categories:

CATEGORIES:
- ShapeMismatch: Errors related to array shape mismatches, dimension errors, broadcasting failures
- DataLeakage: Errors caused by incorrect data handling where train/test contamination or target leakage occurred
- ImportError: Module not found, cannot import name, or missing package errors
- ValueError: Invalid value errors (e.g., NaN/inf in computation, invalid parameter values)
- KeyError: Missing key in dictionary or DataFrame column not found
- TypeError: Incorrect type operations, type mismatches in function arguments
- AttributeError: Missing attribute on an object (e.g., DataFrame has no attribute)
- IndexError: Index out of bounds errors
- Timeout: Execution timed out
- Other: Any error that doesn't fit the above categories

ERROR MESSAGE:
```
{error_message}
```

STDERR OUTPUT:
```
{stderr}
```

CODE THAT FAILED:
```python
{code}
```

Respond in JSON format:
```json
{{
  "error_type": "category name from the list above",
  "confidence": "high/medium/low",
  "suggestion": "Brief suggestion for how to fix this type of error"
}}
```"""
