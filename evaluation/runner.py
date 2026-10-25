import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml

from judge.client import JudgeClient

ROOT = Path(__file__).resolve().parents[1]
LOGS, REPORTS = ROOT / "logs", ROOT / "reports"
LOGS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def load_suite(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Suite file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in [".yaml", ".yml"]:
        return yaml.safe_load(text)
    return json.loads(text)


def log_record(record: dict):
    with (LOGS / "judge_calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_cohens_kappa(judge_verdicts: list[str], gold_verdicts: list[str]) -> dict:
    if not judge_verdicts or len(judge_verdicts) != len(gold_verdicts):
        return {"agreement_rate": None, "cohens_kappa": None}

    n = len(judge_verdicts)
    agreed = sum(j == g for j, g in zip(judge_verdicts, gold_verdicts))
    po = agreed / n

    # Marginals
    labels = sorted(list(set(judge_verdicts) | set(gold_verdicts)))
    pe = 0.0
    for label in labels:
        p_judge = sum(j == label for j in judge_verdicts) / n
        p_gold = sum(g == label for g in gold_verdicts) / n
        pe += p_judge * p_gold

    if pe >= 1.0 or math.isclose(1.0, pe):
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "sample_size": n,
        "agreed_cases": agreed,
        "agreement_rate": round(po, 4),
        "chance_agreement_pe": round(pe, 4),
        "cohens_kappa": round(kappa, 4),
    }


def run_pointwise(suite_path: str = "data/test_suite.json", backend: str | None = None, model: str | None = None):
    suite = load_suite(suite_path)
    judge = JudgeClient(backend=backend, model=model)
    results = []

    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_thoughts_tokens = 0
    total_tokens = 0

    print(f"\n--- Running Pointwise Evaluation on {len(suite)} cases [Backend: {judge.backend}, Model: {judge.model}] ---")

    for case in suite:
        start = time.perf_counter()
        answer = case.get("model_output", "")
        case_id = case.get("case_id", "unknown")
        try:
            verdict, prompt, raw, usage = judge.pointwise(case, answer)
            elapsed = round((time.perf_counter() - start) * 1000, 2)

            # Accumulate token metrics
            p_tok = usage.get("prompt_token_count", 0) or 0
            c_tok = usage.get("candidates_token_count", 0) or 0
            th_tok = usage.get("thoughts_token_count", 0) or 0
            t_tok = usage.get("total_token_count", p_tok + c_tok + th_tok) or 0

            total_prompt_tokens += p_tok
            total_candidates_tokens += c_tok
            total_thoughts_tokens += th_tok
            total_tokens += t_tok

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": "pointwise",
                "case_id": case_id,
                "model": judge.model,
                "backend": judge.backend,
                "prompt": prompt,
                "raw_response": raw,
                "parsed_verdict": verdict,
                "usage": usage,
                "latency_ms": elapsed,
            }
            log_record(record)
            results.append({
                **verdict,
                "gold_verdict": case.get("gold_verdict"),
                "domain": case.get("domain", "general"),
                "latency_ms": elapsed,
                "usage": usage,
                "status": "ok",
            })
            print(f"[{case_id}] {verdict['verdict']} | Score: {verdict['overall_score']:.2f} ({elapsed} ms)")

        except Exception as exc:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            log_record({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": "pointwise",
                "case_id": case_id,
                "backend": judge.backend,
                "model": judge.model,
                "error": str(exc),
                "latency_ms": elapsed,
            })
            results.append({
                "case_id": case_id,
                "status": "error",
                "error": str(exc),
                "latency_ms": elapsed,
            })
            print(f"[{case_id}] ERROR: {exc}")

    valid = [r for r in results if r.get("status") == "ok"]
    n_valid = len(valid)

    # Suite-level aggregations
    passed_cases = sum(r["verdict"] == "PASS" for r in valid)
    pass_rate = round(passed_cases / n_valid, 4) if n_valid else 0.0
    overall_scores = [r["overall_score"] for r in valid]
    mean_overall_score = round(sum(overall_scores) / n_valid, 3) if n_valid else 0.0

    # Score standard deviation (score clustering measure)
    if n_valid > 1:
        variance = sum((s - mean_overall_score) ** 2 for s in overall_scores) / (n_valid - 1)
        score_std_dev = round(math.sqrt(variance), 3)
    else:
        score_std_dev = 0.0

    # Per-criterion mean score aggregation
    criterion_totals: dict[str, float] = {}
    criterion_counts: dict[str, int] = {}
    for r in valid:
        for score_obj in r.get("scores", []):
            c_name = score_obj["criterion"]
            criterion_totals[c_name] = criterion_totals.get(c_name, 0.0) + score_obj["score"]
            criterion_counts[c_name] = criterion_counts.get(c_name, 0) + 1

    criterion_means = {
        c: round(criterion_totals[c] / criterion_counts[c], 3)
        for c in criterion_totals
    }

    # Gold label agreement (if gold_verdict provided)
    cases_with_gold = [r for r in valid if r.get("gold_verdict")]
    if cases_with_gold:
        j_verdicts = [r["verdict"] for r in cases_with_gold]
        g_verdicts = [r["gold_verdict"] for r in cases_with_gold]
        gold_agreement = compute_cohens_kappa(j_verdicts, g_verdicts)
    else:
        gold_agreement = {"note": "No gold_verdict fields found in suite cases."}

    # Estimated Cost ($0.075 per 1M input tokens, $0.30 per 1M output tokens for Flash class)
    est_cost_usd = round(
        (total_prompt_tokens / 1_000_000) * 0.075 + (total_candidates_tokens / 1_000_000) * 0.30,
        6,
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": judge.backend,
        "model": judge.model,
        "suite_source": str(suite_path),
        "total_cases": len(suite),
        "successful_cases": n_valid,
        "failed_cases": len(suite) - n_valid,
        "metrics": {
            "pass_rate": pass_rate,
            "passed_count": passed_cases,
            "failed_count": n_valid - passed_cases,
            "mean_overall_score": mean_overall_score,
            "score_std_dev": score_std_dev,
            "criterion_means": criterion_means,
            "gold_label_validation": gold_agreement,
        },
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidates_tokens": total_candidates_tokens,
            "thoughts_tokens": total_thoughts_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": est_cost_usd,
        },
        "results": results,
    }

    out_file = REPORTS / "initial_report.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPointwise evaluation complete. Report written to: {out_file}")
    print(f"Pass Rate: {pass_rate * 100:.1f}% | Mean Score: {mean_overall_score:.2f} | Total Tokens: {total_tokens}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pointwise LLM-as-Judge evaluation on a test suite.")
    parser.add_argument("--suite", default="data/test_suite.json", help="Path to test suite (JSON or YAML)")
    parser.add_argument("--backend", default=None, help="Judge backend: 'gemini' or 'mock'")
    parser.add_argument("--model", default=None, help="Gemini model name")
    args = parser.parse_args()

    run_pointwise(suite_path=args.suite, backend=args.backend, model=args.model)
