"""
Batch evaluation harness - the data-driven evidence that the agent works.

It runs the full pipeline (plan -> evaluate) over a set of destination cities,
then performs a discrimination test: it feeds the SAME critic a deliberately
degraded plan and confirms the evaluator fails it. A trustworthy evaluation
framework must reject bad output, not just rubber-stamp everything.

Results are printed as a table and written to `eval_results.json` so the numbers
can be quoted in the blog post.

Usage:
    python run_evaluation.py
"""

from __future__ import annotations

import json
import statistics

import config
from tools import get_weather_forecast
from agent import plan_trip
from evaluation import evaluate_plan

# Geographically varied so the mock provider yields a spread of conditions.
SCENARIO_CITIES = [
    "Istanbul", "London", "Barcelona", "Reykjavik", "Dubai",
    "Tokyo", "Cape Town", "Oslo",
]


def _degraded_plan(city: str) -> str:
    """A deliberately bad plan: no warnings, no real activities, no structure.
    The evaluator SHOULD fail this."""
    return (f"Just go to {city}, it will probably be fine. "
            f"Every day looks the same so pick whatever you like.")


def main() -> dict:
    print(f"Runtime mode: {config.mode_label()}\n")
    print(f"{'City':<14}{'BestDay':<11}{'Acc':>5}{'Use':>5}"
          f"{'Saf':>5}{'Coh':>5}{'Overall':>9}{'Pass':>7}")
    print("-" * 66)

    good_rows = []
    per_city = []
    for city in SCENARIO_CITIES:
        result = plan_trip(city, days=config.MAX_FORECAST_DAYS)
        ev = evaluate_plan(result.forecast, result.plan_markdown)
        best = max(result.ranked_days, key=lambda r: r.score)
        good_rows.append(ev)
        per_city.append({
            "city": city,
            "best_day": best.day.day_name,
            "scores": ev.to_dict(),
        })
        print(f"{city:<14}{best.day.day_name:<11}{ev.accuracy:>5.1f}"
              f"{ev.usefulness:>5.1f}{ev.safety:>5.1f}{ev.coherence:>5.1f}"
              f"{ev.overall:>9.2f}{('PASS' if ev.passed else 'FAIL'):>7}")

    # --- Discrimination test ------------------------------------------------
    print("\nDiscrimination test (the evaluator must FAIL bad plans):")
    print(f"{'City':<14}{'Overall':>9}{'Pass':>7}")
    print("-" * 30)
    bad_rows = []
    for city in SCENARIO_CITIES[:4]:
        forecast = get_weather_forecast(city, max_days=config.MAX_FORECAST_DAYS)
        ev_bad = evaluate_plan(forecast, _degraded_plan(city))
        bad_rows.append(ev_bad)
        print(f"{city:<14}{ev_bad.overall:>9.2f}"
              f"{('PASS' if ev_bad.passed else 'FAIL'):>7}")

    # --- Aggregates ---------------------------------------------------------
    good_pass_rate = sum(r.passed for r in good_rows) / len(good_rows)
    bad_pass_rate = sum(r.passed for r in bad_rows) / len(bad_rows)
    summary = {
        "mode": config.mode_label(),
        "evaluator": good_rows[0].evaluator,
        "n_scenarios": len(good_rows),
        "good_plan_pass_rate": round(good_pass_rate, 3),
        "degraded_plan_pass_rate": round(bad_pass_rate, 3),
        "mean_accuracy": round(statistics.mean(r.accuracy for r in good_rows), 2),
        "mean_usefulness": round(statistics.mean(r.usefulness for r in good_rows), 2),
        "mean_safety": round(statistics.mean(r.safety for r in good_rows), 2),
        "mean_coherence": round(statistics.mean(r.coherence for r in good_rows), 2),
        "mean_overall": round(statistics.mean(r.overall for r in good_rows), 2),
        "per_city": per_city,
    }

    print("\n=== Aggregate results ===")
    print(f"Evaluator              : {summary['evaluator']}")
    print(f"Scenarios              : {summary['n_scenarios']}")
    print(f"Good-plan pass rate    : {good_pass_rate:.0%}")
    print(f"Degraded-plan pass rate: {bad_pass_rate:.0%}  (lower is better)")
    print(f"Mean overall score     : {summary['mean_overall']} / 5")
    print(f"  accuracy={summary['mean_accuracy']}  usefulness={summary['mean_usefulness']}"
          f"  safety={summary['mean_safety']}  coherence={summary['mean_coherence']}")

    with open("eval_results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\nWrote eval_results.json")
    return summary


if __name__ == "__main__":
    main()
