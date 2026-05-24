"""Prompt templates for ablation phase: ABLATION_STUDY_PROMPT, ABLATION_SUMMARIZE_PROMPT, EXTRACTOR_PROMPT.

Used by nodes/ablation.py (A4, A5, A6) to generate ablation studies,
summarize results, and extract the target block for refinement.
"""

ABLATION_STUDY_PROMPT = """You are an expert ML engineer conducting an ablation study on a machine learning pipeline.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric} ({direction})

CURRENT SOLUTION:
```python
{solution}
```

FUNCTIONAL BLOCKS IDENTIFIED:
{functional_blocks}

PREVIOUS ABLATION SUMMARIES (avoid repeating these):
{previous_summaries}

INSTRUCTIONS:
1. For each functional block, create an ablation variant that DISABLES or REMOVES that block while keeping the rest of the pipeline intact.
2. Each variant should be a COMPLETE, self-contained Python script.
3. The baseline script is the original solution unmodified.
4. Each ablation variant should comment out or replace the target block with a minimal passthrough/no-op.
5. All scripts must read train.csv and test.csv from the current directory.
6. All scripts must use 5-fold cross-validation and print: Final Validation Performance: <score>
7. For multi-target prediction, handle each target appropriately.

Return a JSON object with the following format (return ONLY valid JSON, do NOT wrap in markdown code fences):
{{
  "ablation_scripts": [
    {{
      "name": "baseline",
      "block_name": "baseline",
      "code": "complete baseline script here"
    }},
    {{
      "name": "ablation_<block_name>",
      "block_name": "<block_name>",
      "code": "complete script with that block disabled here"
    }}
  ]
}}

Generate ablation scripts for the 2-3 most impactful blocks. Focus on blocks NOT previously studied."""

ABLATION_SUMMARIZE_PROMPT = """You are analyzing ablation study results for a machine learning pipeline.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

ABLATION RESULTS:
{ablation_results}

INSTRUCTIONS:
Analyze the ablation results which show the performance impact of disabling each component.
- Compare each variant's score against the baseline to determine impact.
- Identify which components are most critical for performance.
- Provide recommendations for which block to focus refinement on.

Produce a clear, structured summary that:
1. Lists each component and its impact score (baseline_score - variant_score)
2. Ranks components by impact (highest impact first)
3. Recommends the most impactful block for refinement
4. Notes any surprising findings

Format your summary as plain text, clearly structured."""

EXTRACTOR_PROMPT = """You are an expert ML code analyst. Your job is to identify the most impactful code block to refine next.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

ABLATION SUMMARY:
{ablation_summary}

CURRENT SOLUTION:
```python
{solution}
```

FUNCTIONAL BLOCKS:
{functional_blocks}

PREVIOUSLY REFINED BLOCKS (do NOT select these again):
{previous_blocks}

INSTRUCTIONS:
1. Based on the ablation summary and functional blocks, select the single most impactful block to refine next.
2. Do NOT select any block that has already been refined (listed above).
3. Extract the EXACT code for that block from the solution — it must be an exact substring.
4. Draft an initial improvement plan (p_0) describing specific improvements for this block.

Return a JSON object (return ONLY valid JSON, do NOT wrap in markdown code fences):
{{
  "target_block_name": "name of the selected block",
  "target_block_code": "exact code from the solution for this block",
  "initial_plan": "detailed improvement plan for this block"
}}"""
