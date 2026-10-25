import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from judge.client import JudgeClient
from evaluation.runner import ROOT, load_suite, log_record, REPORTS, compute_cohens_kappa


def compare_ab(suite_path: str = "data/test_suite.json", backend: str | None = None, model: str | None = None):
    suite = load_suite(suite_path)
    judge = JudgeClient(backend=backend, model=model)

    print(f"\n--- Running A/B Config Comparison with Position Debiasing on {len(suite)} cases ---")
    print(f"Backend: {judge.backend} | Model: {judge.model}\n")

    rows = []
    total_tokens = 0
    total_prompt_tokens = 0
    total_candidates_tokens = 0

    for case in suite:
        cid = case["case_id"]
        a = case.get("config_a_output", case.get("model_output", ""))
        b = case.get("config_b_output", case.get("model_output", ""))
        gen_a = case.get("generator_a_family", "unknown")
        gen_b = case.get("generator_b_family", "unknown")

        start = time.perf_counter()
        try:
            # 1. Pointwise scoring for Config A
            va, pa, ra, ua = judge.pointwise(case, a, f"{cid}_A")
            # 2. Pointwise scoring for Config B
            vb, pb, rb, ub = judge.pointwise(case, b, f"{cid}_B")

            # 3. Pairwise Order 1: A then B
            pair_ab, pab, rab, uab = judge.pairwise(case, a, b, "A_then_B")
            # 4. Pairwise Order 2: B then A (Mitigating position bias)
            pair_ba, pba, rba, uba = judge.pairwise(case, b, a, "B_then_A")

            # Map Order 2 raw winner back to original candidate labels
            winner_ab = pair_ab.get("winner", "TIE")
            raw_winner_ba = pair_ba.get("winner", "TIE")
            winner_ba_mapped = {"A": "B", "B": "A", "TIE": "TIE"}.get(raw_winner_ba, "INVALID")

            # Check for position flip
            is_tie_in_either = (winner_ab == "TIE" or winner_ba_mapped == "TIE")
            flip = (winner_ab != winner_ba_mapped) if not is_tie_in_either else False

            # Debiased consensus winner: require agreement between both presentations
            if winner_ab == winner_ba_mapped:
                consensus_winner = winner_ab
            else:
                consensus_winner = "TIE (POSITION_FLIP)"

            elapsed = round((time.perf_counter() - start) * 1000, 2)

            for u in (ua, ub, uab, uba):
                p_tok = u.get("prompt_token_count", 0) or 0
                c_tok = u.get("candidates_token_count", 0) or 0
                t_tok = u.get("total_token_count", p_tok + c_tok) or 0
                total_prompt_tokens += p_tok
                total_candidates_tokens += c_tok
                total_tokens += t_tok

            # Audit logs for all 4 calls
            for task, prompt, raw, usage in [
                ("ab_pointwise_A", pa, ra, ua),
                ("ab_pointwise_B", pb, rb, ub),
                ("ab_pairwise_AB", pab, rab, uab),
                ("ab_pairwise_BA", pba, rba, uba),
            ]:
                log_record({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task": task,
                    "case_id": cid,
                    "model": judge.model,
                    "backend": judge.backend,
                    "prompt": prompt,
                    "raw_response": raw,
                    "usage": usage,
                })

            row = {
                "case_id": cid,
                "status": "ok",
                "generator_a_family": gen_a,
                "generator_b_family": gen_b,
                "gold_pairwise_winner": case.get("gold_pairwise_winner"),
                "config_A": va,
                "config_B": vb,
                "order_1_A_then_B": {
                    "winner": winner_ab,
                    "confidence": pair_ab.get("confidence"),
                    "rationale": pair_ab.get("rationale"),
                },
                "order_2_B_then_A": {
                    "raw_winner": raw_winner_ba,
                    "mapped_to_original_labels": winner_ba_mapped,
                    "confidence": pair_ba.get("confidence"),
                    "rationale": pair_ba.get("rationale"),
                },
                "position_flip": flip,
                "debiased_consensus_winner": consensus_winner,
                "latency_ms": elapsed,
            }
            rows.append(row)
            print(
                f"[{cid}] A_Score={va['overall_score']:.1f} | B_Score={vb['overall_score']:.1f} | "
                f"Order1={winner_ab} Order2={winner_ba_mapped} -> Consensus: {consensus_winner}"
            )

        except Exception as exc:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            log_record({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": "ab_comparison_error",
                "case_id": cid,
                "error": str(exc),
                "latency_ms": elapsed,
            })
            rows.append({
                "case_id": cid,
                "status": "error",
                "error": str(exc),
                "latency_ms": elapsed,
            })
            print(f"[{cid}] ERROR: {exc}")

    valid = [r for r in rows if r.get("status") == "ok"]
    n_valid = len(valid)

    # Pointwise statistics
    passed_a = sum(r["config_A"]["verdict"] == "PASS" for r in valid)
    passed_b = sum(r["config_B"]["verdict"] == "PASS" for r in valid)
    scores_a = [r["config_A"]["overall_score"] for r in valid]
    scores_b = [r["config_B"]["overall_score"] for r in valid]

    # Single-order vs. Debiased pairwise metrics
    raw_wins_a = sum(r["order_1_A_then_B"]["winner"] == "A" for r in valid)
    raw_wins_b = sum(r["order_1_A_then_B"]["winner"] == "B" for r in valid)
    raw_ties = sum(r["order_1_A_then_B"]["winner"] == "TIE" for r in valid)

    debiased_wins_a = sum(r["debiased_consensus_winner"] == "A" for r in valid)
    debiased_wins_b = sum(r["debiased_consensus_winner"] == "B" for r in valid)
    debiased_ties = sum("TIE" in r["debiased_consensus_winner"] for r in valid)
    position_flips = sum(r.get("position_flip", False) for r in valid)

    decisive_debiased = debiased_wins_a + debiased_wins_b

    # Overall winner declaration
    if n_valid > 0:
        if debiased_wins_a > debiased_wins_b:
            declared_winner = "Config A"
        elif debiased_wins_b > debiased_wins_a:
            declared_winner = "Config B"
        else:
            declared_winner = "TIE (Inconclusive)"
    else:
        declared_winner = "INCONCLUSIVE (No successful cases)"

    # Gold agreement on pairwise decisions if present
    gold_pairwise_cases = [r for r in valid if r.get("gold_pairwise_winner")]
    if gold_pairwise_cases:
        j_winners = [
            "A" if r["debiased_consensus_winner"] == "A" else ("B" if r["debiased_consensus_winner"] == "B" else "TIE")
            for r in gold_pairwise_cases
        ]
        g_winners = [r["gold_pairwise_winner"] for r in gold_pairwise_cases]
        gold_pairwise_agreement = compute_cohens_kappa(j_winners, g_winners)
    else:
        gold_pairwise_agreement = {"note": "No gold_pairwise_winner fields in suite cases."}

    est_cost_usd = round(
        (total_prompt_tokens / 1_000_000) * 0.075 + (total_candidates_tokens / 1_000_000) * 0.30,
        6,
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": judge.backend,
        "model": judge.model,
        "suite_source": str(suite_path),
        "total_cases": len(suite),
        "successful_cases": n_valid,
        "failed_cases": len(suite) - n_valid,
        "config_A_pointwise": {
            "pass_rate": round(passed_a / n_valid, 4) if n_valid else None,
            "mean_score": round(sum(scores_a) / n_valid, 3) if n_valid else None,
        },
        "config_B_pointwise": {
            "pass_rate": round(passed_b / n_valid, 4) if n_valid else None,
            "mean_score": round(sum(scores_b) / n_valid, 3) if n_valid else None,
        },
        "raw_single_order_pairwise": {
            "A_wins": raw_wins_a,
            "B_wins": raw_wins_b,
            "ties": raw_ties,
        },
        "debiased_pairwise_mitigation": {
            "method": "Order-swap with required agreement (both A_then_B and B_then_A)",
            "position_flips_detected": position_flips,
            "position_flip_rate": round(position_flips / n_valid, 4) if n_valid else 0.0,
            "consensus_A_wins": debiased_wins_a,
            "consensus_B_wins": debiased_wins_b,
            "consensus_ties_or_flips": debiased_ties,
            "A_win_rate_excluding_ties": round(debiased_wins_a / decisive_debiased, 4) if decisive_debiased else None,
        },
        "gold_pairwise_agreement": gold_pairwise_agreement,
        "declared_winner": declared_winner,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidates_tokens": total_candidates_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": est_cost_usd,
        },
        "cases": rows,
    }

    out_file = REPORTS / "ab_comparison.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nA/B Comparison complete. Report written to: {out_file}")
    print(f"Declared Winner: {declared_winner} (A Wins: {debiased_wins_a}, B Wins: {debiased_wins_b}, Ties/Flips: {debiased_ties})")
    print(f"Position Flip Rate: {summary['debiased_pairwise_mitigation']['position_flip_rate'] * 100:.1f}%\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A/B configuration comparison with position debiasing.")
    parser.add_argument("--suite", default="data/test_suite.json", help="Path to test suite (JSON or YAML)")
    parser.add_argument("--backend", default=None, help="Judge backend: 'gemini' or 'mock'")
    parser.add_argument("--model", default=None, help="Gemini model name")
    args = parser.parse_args()

    compare_ab(suite_path=args.suite, backend=args.backend, model=args.model)
