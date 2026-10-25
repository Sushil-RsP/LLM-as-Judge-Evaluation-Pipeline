# Submission Notes — Problem 2: LLM-as-Judge Evaluation Pipeline

This document provides a concise technical summary and verification checklist mapped directly to the assignment evaluation rubric (100 Pts).

---

## 1. Architecture & Pipeline Correctness (20 Pts)
- **Input Flexibility**: Ingests test suites in either **JSON** or **YAML** (`data/test_suite.json`, `data/test_suite.yaml`) using dynamic schema loaders.
- **Structured Pydantic Schemas**: [`judge/schemas.py`](judge/schemas.py) defines `CriterionScore`, `PointwiseVerdict`, and `PairwiseVerdict` with strict validation (`extra="forbid"`).
- **Robust JSON Handling**: [`judge/client.py`](judge/client.py) includes `parse_json_robust()` which strips markdown fences, repairs trailing commas, and extracts valid JSON objects even if surrounded by commentary.
- **Auditable Logging**: Every prompt, raw response, parsed output, token usage count, and latency is preserved in [`logs/judge_calls.jsonl`](logs/judge_calls.jsonl).

---

## 2. Judging Design & Explicit Rubrics (20 Pts)
- **Implemented Modes**: Pointwise scoring, Pairwise A-vs-B comparison, Reference-based evaluation, and Reference-free evaluation. Documented in [`README.md`](README.md) with operational guidance on when each fits.
- **Explicit 6-Criterion Rubric**:
  - `correctness`: Factual accuracy against reference output.
  - `faithfulness`: Strict support from context without hallucination.
  - `completeness`: Thoroughness without irrelevant padding.
  - `instruction_following`: Exact adherence to prompt formatting and constraints.
  - `tone`: Objective, professional tone without sycophancy or aggression.
  - `safety`: Full compliance with safety and ethical standards.
- **Deterministic Verdict**: Overall score is mathematically computed as the mean of criteria. A PASS verdict strictly requires `overall_score >= 4.0` and `correctness >= 4.0`.

---

## 3. Bias Handling: Named, Mitigated in Code, & Measured (25 Pts)
- **Position Bias (A/B Order)**:
  - *Mitigated in code:* [`evaluation/compare_ab.py`](evaluation/compare_ab.py) executes both `A_then_B` and `B_then_A`. Winner is declared only if both presentations agree; disagreeing pairs are designated as `TIE (POSITION_FLIP)`.
  - *Measured:* `position_flip_rate` is tracked in both `ab_comparison.json` and `bias_validation.json`.
- **Verbosity / Length Bias**:
  - *Mitigated in code:* Rubric explicitly penalizes ungrounded padding under `completeness`. Tested with synthetic padded fluff probes (8 repetitive sentences).
  - *Measured:* `padded_fluff_preference_rate` in `bias_validation.json`.
- **Self-Enhancement Bias**:
  - *Mitigated in code:* Test suite cases specify generator families (`gpt`, `claude`, `llama`). Judge family (`gemini`) is audited against generator family in `bias_validation.json` to ensure cross-family isolation.
- **Sycophancy & Style Bias**:
  - *Mitigated in code:* Mandated verbatim evidence citation in JSON verdicts. Tested with 5 adversarial probes (`verbose_wrong`, `confidently_wrong`, `terse_correct`, `sycophantic_flattery`, `hallucinated_citations`).
  - *Measured:* `adversarial_robustness_rate` in `bias_validation.json`.
- **Score Clustering**:
  - *Mitigated in code:* Concrete 1–5 anchors calibrated in prompt.
  - *Measured:* `score_clustering_std_dev` measures score variance across the suite.

---

## 4. Judge Validation (20 Pts)
- **Human / Gold Label Agreement**:
  - Calculated using **Cohen's Kappa ($\kappa$)** and agreement rate between judge verdicts and gold verdicts (`gold_verdict`, `gold_pairwise_winner`).
- **Test-Retest Consistency**:
  - Measured across repeated trials under zero temperature to record verdict flip rates and score standard deviation.
- **Adversarial Robustness**:
  - Evaluated on 5 diagnostic probes in [`data/adversarial_probes.json`](data/adversarial_probes.json).

---

## 5. Comparison & Engineering (15 Pts)
- **A/B Winner Call**: Declares winner between Config A and Config B with pass rates, mean scores, decisive win rates, and debiased consensus.
- **Cost & Token Tracking**: Aggregated prompt, completion, and total tokens tracked with USD cost estimation.
- **Reproducibility**: Clear setup, dual-backend support (Gemini API and deterministic Mock mode), and comprehensive discussion.
