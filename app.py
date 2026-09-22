"""
Gradio web interface for the Weather-Aware Travel Planning Agent.

This is the Hugging Face Spaces entry point (`app_file: app.py`). The core
handler `run_planner()` is importable and callable without launching a server,
which is what the test suite and the evaluation harness use.

Run locally:   python app.py
Deploy:        push this repo to a Hugging Face Space (SDK = gradio); see DEPLOYMENT_GUIDE.md.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend for servers / Spaces
import matplotlib.pyplot as plt

import gradio as gr

import config
from agent import plan_trip
from evaluation import evaluate_plan


def _weather_table(ranked_days) -> list[list]:
    rows = []
    for r in ranked_days:
        d = r.day
        rows.append([
            d.day_name, d.date, d.description.capitalize(),
            f"{d.temp_min:.0f}/{d.temp_max:.0f}",
            f"{int(d.precip_prob * 100)}%",
            f"{d.wind_kph:.0f}",
            f"{r.score:.0f}", r.verdict,
        ])
    return rows


def _weather_chart(ranked_days):
    days = [r.day for r in ranked_days]
    labels = [f"{d.day_name[:3]}\n{d.date[5:]}" for d in days]
    tmin = [d.temp_min for d in days]
    tmax = [d.temp_max for d in days]
    precip = [d.precip_prob * 100 for d in days]
    scores = [r.score for r in ranked_days]

    fig, ax1 = plt.subplots(figsize=(8, 3.6))
    x = range(len(days))

    # Temperature range as a vertical band per day.
    ax1.fill_between(x, tmin, tmax, alpha=0.25, color="#e07b39",
                     label="Temp range (C)")
    ax1.plot(x, [ (a + b) / 2 for a, b in zip(tmin, tmax) ],
             color="#e07b39", marker="o", label="Avg temp (C)")
    ax1.set_ylabel("Temperature (C)", color="#e07b39")
    ax1.tick_params(axis="y", labelcolor="#e07b39")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)

    # Precipitation probability + suitability score on a second axis.
    ax2 = ax1.twinx()
    ax2.bar([i + 0.0 for i in x], precip, width=0.35, alpha=0.5,
            color="#3b82c4", label="Rain prob (%)")
    ax2.plot(x, scores, color="#2e9e5b", marker="s", linewidth=2,
             label="Suitability (0-100)")
    ax2.set_ylabel("Rain % / Suitability", color="#333")
    ax2.set_ylim(0, 105)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def run_planner(city: str, days: int, preferences: str):
    """Core handler. Returns the six UI outputs. Safe to call directly."""
    city = (city or "").strip()
    if not city:
        return ("**Please enter a destination city.**", "", [], None, "", "")

    result = plan_trip(city, days=int(days), preferences=preferences)
    ev = evaluate_plan(result.forecast, result.plan_markdown)

    mode_md = (f"**Mode:** {result.mode}  \n"
               f"**Forecast source:** {result.forecast.source}  ")

    table = _weather_table(result.ranked_days)
    chart = _weather_chart(result.ranked_days)

    eval_md = (
        f"### Evaluation ({ev.evaluator})\n"
        f"| Dimension | Score (1-5) |\n|---|---|\n"
        f"| Accuracy | {ev.accuracy} |\n"
        f"| Usefulness | {ev.usefulness} |\n"
        f"| Safety | {ev.safety} |\n"
        f"| Coherence | {ev.coherence} |\n"
        f"| **Overall** | **{ev.overall}** |\n\n"
        f"**Verdict:** {'✅ PASS' if ev.passed else '❌ FAIL'}  \n"
        f"**Feedback:** {ev.feedback}"
    )

    trace = result.reasoning_trace
    return mode_md, result.plan_markdown, table, chart, trace, eval_md


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Weather-Aware Travel Planning Agent",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🌤️ Weather-Aware Travel Planning Agent\n"
            "An autonomous agent that reads the weather forecast for your "
            "destination, ranks the best days to visit, recommends "
            "weather-appropriate activities, and is graded by an independent "
            "LLM critic.\n\n"
            "*SEN4018 Semester-Long Project - Bahcesehir University.*"
        )
        mode_banner = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1):
                city = gr.Textbox(label="Destination city",
                                  placeholder="e.g. Istanbul", value="Istanbul")
                days = gr.Slider(1, config.MAX_FORECAST_DAYS, value=5, step=1,
                                 label="Days to consider (forecast horizon)")
                prefs = gr.Textbox(
                    label="Preferences (optional)",
                    placeholder="e.g. museums, food, outdoor, family-friendly")
                go = gr.Button("Plan my trip", variant="primary")
                gr.Examples(
                    examples=[["Istanbul", 5, "museums, food"],
                              ["London", 4, "outdoor, parks"],
                              ["Barcelona", 5, "beach, sightseeing"]],
                    inputs=[city, days, prefs],
                )
            with gr.Column(scale=2):
                plan_out = gr.Markdown(label="Travel plan")

        with gr.Row():
            chart_out = gr.Plot(label="Forecast & suitability")
        with gr.Row():
            table_out = gr.Dataframe(
                headers=["Day", "Date", "Condition", "Min/Max C",
                         "Rain%", "Wind kph", "Score", "Verdict"],
                label="Daily forecast & suitability", wrap=True)
        with gr.Accordion("Agent reasoning trace (tool calls)", open=False):
            trace_out = gr.Textbox(label="", lines=10)
        eval_out = gr.Markdown()

        go.click(run_planner, inputs=[city, days, prefs],
                 outputs=[mode_banner, plan_out, table_out, chart_out,
                          trace_out, eval_out])
        demo.load(lambda: f"**Mode:** {config.mode_label()}", outputs=mode_banner)
    return demo


if __name__ == "__main__":
    build_demo().launch()
