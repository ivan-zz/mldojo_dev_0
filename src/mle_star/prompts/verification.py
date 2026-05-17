"""Prompt templates for verification: SEMANTIC_VERIFY_PROMPT, SAST_CHECK_PROMPT.

Used by nodes/verification.py (A_verify, A_sast) to check semantic
correctness and security of refined code before execution.
"""

SEMANTIC_VERIFY_PROMPT = """You are an expert ML code reviewer. Verify the semantic correctness of the following refined code block.

TASK DESCRIPTION:
{task_desc}

EVALUATION METRIC: {metric}

ORIGINAL FULL SOLUTION:
```python
{current_solution}
```

REFINED BLOCK:
```python
{refined_code}
```

Perform these checks and report pass/fail for each:

1. TASK ALIGNMENT: Does the refined block address the correct task described above, not a different task?
2. TARGET VARIABLE: Does it use the correct target variable(s) for this task?
3. METRIC COMPUTATION: Is the evaluation metric ({metric}) computed properly and consistently with the original solution?
4. DATA SPLITTING: Does the data splitting strategy make sense (proper train/validation separation, no leakage in splitting)?
5. INTERFACE COMPATIBILITY: Is the function signature compatible with the original block (same inputs/outputs)?
6. DATA SOURCES: Does it use only allowed data sources (train.csv, test.csv) and not external files?
7. LOGICAL CORRECTNESS: Are there any obvious logical errors or contradictions with the original solution?

Respond in JSON format:
```json
{{
  "status": "ok" or "semantic_fail",
  "feedback": "If semantic_fail, explain what is wrong and how to fix it. If ok, leave empty string.",
  "failed_checks": ["list of check names that failed, e.g. 'target_variable', 'metric_computation'"],
  "checks": {{
    "task_alignment": "pass" or "fail",
    "target_variable": "pass" or "fail",
    "metric_computation": "pass" or "fail",
    "data_splitting": "pass" or "fail",
    "interface_compatibility": "pass" or "fail",
    "data_sources": "pass" or "fail",
    "logical_correctness": "pass" or "fail"
  }}
}}
```"""

SAST_CHECK_PROMPT = """You are an expert security code reviewer. Perform a deep security analysis of the following Python code.

TASK DESCRIPTION:
{task_desc}

CODE TO ANALYZE:
```python
{code}
```

PRELIMINARY FINDINGS (from automated checks):
{preliminary_findings}

Check for these security issues:

1. COMMAND INJECTION: Any os.system, subprocess with shell=True, or similar patterns that could execute arbitrary commands.
2. UNSAFE DESERIALIZATION: pickle.loads, pickle.load, or similar that could execute arbitrary code.
3. HALLUCINATED IMPORTS: Import statements for packages that don't exist (the LLM may have fabricated them).
4. DANGEROUS DYNAMIC CODE: eval(), exec(), compile(), __import__() that could run arbitrary code.
5. NETWORK ACCESS: socket, http, urllib, requests that access external resources (should be blocked in sandbox).
6. FILE SYSTEM MANIPULATION: shutil.rmtree, Path.unlink, or other destructive file operations.
7. DATA EXFILTRATION: Code that could read sensitive files or send data externally.
8. RESOURCE EXHAUSTION: Infinite loops, unbounded memory allocation, or fork bombs.

For each issue, classify severity:
- critical: Must be blocked before execution
- warning: Should be flagged but can proceed

Respond in JSON format:
```json
{{
  "status": "pass" or "critical_violation",
  "violations": [
    {{
      "type": "violation_type (e.g. command_injection, unsafe_deserialization)",
      "description": "Description of the violation",
      "line": line_number_or_range,
      "severity": "critical" or "warning",
      "cwe": "CWE-ID if applicable (e.g. CWE-78)"
    }}
  ],
  "report": "Human-readable summary of findings"
}}
```"""
