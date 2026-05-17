"""Prompt templates for search phase: RETRIEVER_PROMPT, CANDIDATE_EVAL_PROMPT, MERGER_PROMPT.

These prompts are used by the search phase nodes (A1__retrieve, A2__generate, A3__merge)
to interact with the LLM. Each function returns a formatted prompt string.
"""

RETRIEVER_PROMPT = """You are an expert ML engineer helping to select the best models for a regression task.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

CONSTRAINT: Only suggest models available in scikit-learn (sklearn). Do NOT suggest XGBoost, LightGBM, CatBoost, PyTorch, TensorFlow, or any library outside scikit-learn.

Based on the task description, suggest {num_models} diverse and suitable machine learning models. For each model, provide:
1. **model_name**: The exact sklearn class name (e.g., "RandomForestRegressor")
2. **description**: Brief explanation (2-3 sentences) of why this model is suitable for this task
3. **example_code**: A minimal sklearn code snippet showing how to instantiate and train this model

Focus on models that are likely to perform well on tabular/structured data regression tasks. Prefer a diverse set covering different approaches (tree-based, linear, ensemble, kernel-based).

Respond in JSON format:
```json
{{
  "models": [
    {{
      "model_name": "SklearnClassName",
      "description": "Why this model is suitable...",
      "example_code": "from sklearn.module import ClassName\\nmodel = ClassName(...)"
    }}
  ]
}}
```"""

CANDIDATE_EVAL_PROMPT = """You are an expert ML engineer. Generate a complete, working Python script for the following ML task.

TASK DESCRIPTION:
{task_desc}

MODEL: {model_name}
MODEL DESCRIPTION: {model_description}

EVALUATION METRIC: {metric} ({direction})

DATASET INFO:
- Training CSV: train.csv (columns: {feature_cols})
- Target columns: {target_cols}
- Test CSV: test.csv (same features, no targets)
- The script should read data from the current directory

REQUIREMENTS:
1. Import only sklearn and standard libraries (numpy, pandas). No xgboost, lightgbm, etc.
2. Read train.csv and test.csv from the current directory
3. Train a {model_name} model
4. Use 5-fold cross-validation to evaluate using the {metric} metric
5. Print ONLY the final score as: Final Validation Performance: <score>
6. Do NOT use any external data files beyond train.csv and test.csv
7. Handle missing values appropriately
8. Feature engineering is allowed and encouraged

For multi-target prediction targets, train separate models for each target and average the scores.

The script must be a COMPLETE, self-contained Python file that can be run directly.
Start with imports and end with printing the final score.

{additional_constraints}"""

MERGER_PROMPT = """You are an expert ML engineer. Merge the following two ML pipeline scripts into a single improved ensemble script.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric} ({direction})

BASE SCRIPT (better performer):
```python
{base_code}
```

REFERENCE SCRIPT:
```python
{ref_code}
```

REQUIREMENTS FOR THE MERGED SCRIPT:
1. Combine the best ideas from both scripts (e.g., different feature engineering, model averaging, stacking)
2. Use only sklearn and standard libraries (numpy, pandas)
3. Read train.csv and test.csv from the current directory
4. Use 5-fold cross-validation with the {metric} metric
5. Print ONLY the final score as: Final Validation Performance: <score>
6. For multi-target prediction, handle each target appropriately
7. The merged script should be a COMPLETE, self-contained Python file
8. Aim to improve upon the base script's performance by leveraging both approaches

Focus on genuine improvements: better feature engineering, model averaging, or combining different model strengths. Do NOT simply copy the base script."""
