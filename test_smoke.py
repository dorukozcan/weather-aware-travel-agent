"""
Fast smoke tests that run with NO API keys (demo mode).

    python test_smoke.py

Verifies the full pipeline end to end: tools -> agent loop -> evaluation -> UI
handler, plus the evaluator's ability to reject a degraded plan.
"""

from __future__ import annotations

import config
from tools import get_weather_forecast, suggest_activities
from agent import plan_trip
from evaluation import evaluate_plan, heuristic_evaluate


def test_forecast_shape():
    fc = get_weather_forecast("Istanbul", max_days=5)
    assert fc.days, "forecast should not be empty"
    assert len(fc.days) == 5
    d = fc.days[0]
    assert d.temp_min <= d.temp_max
    assert 0.0 <= d.precip_prob <= 1.0
    print("ok  test_forecast_shape")


def test_determinism():
    a = get_weather_forecast("Tokyo", max_days=5).as_text()
    b = get_weather_forecast("Tokyo", max_days=5).as_text()
    assert a == b, "mock forecast must be deterministic within a day"
    print("ok  test_determinism")


def test_activities():
    res = suggest_activities("Rain", 14, "museums")
    assert res["category"] == "indoor"
    assert res["activities"]
    print("ok  test_activities")


def test_plan_and_eval():
    result = plan_trip("Barcelona", days=5, preferences="food, outdoor")
    assert result.plan_markdown
    assert result.ranked_days and len(result.ranked_days) == 5
    assert result.tool_calls, "agent must record tool usage"
    assert "Travel plan" in result.plan_markdown
    ev = evaluate_plan(result.forecast, result.plan_markdown)
    assert ev.passed, f"a good plan should pass, got {ev.to_dict()}"
    print(f"ok  test_plan_and_eval (overall={ev.overall}, pass={ev.passed})")


def test_evaluator_rejects_garbage():
    fc = get_weather_forecast("London", max_days=5)
    bad = "Go to London. It will be fine."
    ev = heuristic_evaluate(fc, bad)
    assert not ev.passed, "evaluator must reject a degraded plan"
    print(f"ok  test_evaluator_rejects_garbage (overall={ev.overall})")


def test_ui_handler():
    from app import run_planner
    mode, plan, table, chart, trace, eval_md = run_planner("Oslo", 4, "")
    assert plan and table and chart is not None
    assert len(table) == 4
    assert "Evaluation" in eval_md
    print("ok  test_ui_handler")


if __name__ == "__main__":
    print(f"Mode under test: {config.mode_label()}\n")
    test_forecast_shape()
    test_determinism()
    test_activities()
    test_plan_and_eval()
    test_evaluator_rejects_garbage()
    test_ui_handler()
    print("\nAll smoke tests passed.")
