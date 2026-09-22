"""
The agentic core of the project.

This module implements the REQUIRED autonomous decision loop and tool usage.

Two interchangeable engines produce a travel plan from a forecast:

  * _plan_with_llm()   - FULL MODE. A LangChain tool-calling (ReAct-style) agent
                         powered by Groq / LLaMA 3 decides, on its own, which
                         tools to call and in what order, reasons over every
                         forecast day, then writes the plan. Tool calls and the
                         reasoning trace are captured for transparency.

  * _plan_rule_based() - DEMO MODE (no Groq key). A deterministic controller that
                         executes the same decide -> act -> observe loop in pure
                         Python: it scores each day, ranks them, and selects
                         activities through the same tools. This guarantees the
                         app is fully runnable and gradable without any API key.

Both return an identical `PlanResult`, so the UI and the evaluation framework do
not care which engine ran.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import config
from tools import (
    Forecast,
    DayForecast,
    get_weather_forecast,
    suggest_activities,
)


@dataclass
class RankedDay:
    day: DayForecast
    score: float                       # 0..100 travel-suitability
    verdict: str                       # Excellent / Good / Fair / Poor
    activities: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class PlanResult:
    city: str
    mode: str
    plan_markdown: str
    forecast: Forecast
    ranked_days: list = field(default_factory=list)   # list[RankedDay]
    reasoning_trace: str = ""
    tool_calls: list = field(default_factory=list)     # list[str]


# ---------------------------------------------------------------------------
# Shared scoring logic (used by the rule-based engine and by the heuristic
# evaluator). Encapsulates the "is this a good day to travel?" decision.
# ---------------------------------------------------------------------------
def score_day(day: DayForecast) -> tuple[float, str, list]:
    """Return (score 0..100, verdict, warnings) for one forecast day."""
    score = 100.0
    warnings: list[str] = []

    score -= day.precip_prob * 45
    score -= min(day.rain_mm * 2.5, 25)

    if day.condition == "Thunderstorm":
        score -= 35
        warnings.append("Thunderstorms expected - avoid outdoor plans and travel by road.")
    elif day.condition in ("Rain", "Snow"):
        if day.rain_mm >= 5 or day.precip_prob >= 0.6:
            warnings.append(f"{day.condition} likely ({int(day.precip_prob*100)}% / {day.rain_mm:.1f}mm) - pack accordingly.")
    elif day.condition == "Drizzle":
        score -= 5

    # Wind
    if day.wind_kph >= 45:
        score -= 18
        warnings.append(f"Strong wind ({day.wind_kph:.0f} km/h) - exposed activities not advised.")
    elif day.wind_kph >= 30:
        score -= 8

    # Temperature comfort band (15-26 C ideal)
    if day.temp_avg > 33:
        score -= (day.temp_avg - 33) * 2.5
        warnings.append(f"Very hot (avg {day.temp_avg:.0f}C) - risk of heat stress at midday.")
    elif day.temp_avg > 26:
        score -= (day.temp_avg - 26) * 1.2
    elif day.temp_avg < 0:
        score -= (0 - day.temp_avg) * 2.0
        warnings.append(f"Freezing (avg {day.temp_avg:.0f}C) - dress for cold and ice.")
    elif day.temp_avg < 10:
        score -= (10 - day.temp_avg) * 1.0

    score = max(0.0, min(100.0, score))
    if score >= 80:
        verdict = "Excellent"
    elif score >= 60:
        verdict = "Good"
    elif score >= 40:
        verdict = "Fair"
    else:
        verdict = "Poor"
    return round(score, 1), verdict, warnings


# ---------------------------------------------------------------------------
# DEMO MODE engine: deterministic decide -> act -> observe loop
# ---------------------------------------------------------------------------
def _plan_rule_based(forecast: Forecast, preferences: str) -> PlanResult:
    trace_lines = [
        f"[loop] Goal: recommend the best days to visit {forecast.city}.",
        f"[tool] get_weather_forecast('{forecast.city}') -> {len(forecast.days)} days "
        f"(source: {forecast.source}).",
    ]
    tool_calls = [f"get_weather_forecast('{forecast.city}')"]

    ranked: list[RankedDay] = []
    for day in forecast.days:
        score, verdict, warnings = score_day(day)
        act = suggest_activities(day.condition, day.temp_avg, preferences)
        tool_calls.append(
            f"suggest_activities('{day.condition}', {day.temp_avg:.0f})")
        trace_lines.append(
            f"[observe] {day.day_name}: condition={day.condition}, "
            f"avg={day.temp_avg:.0f}C, rain={int(day.precip_prob*100)}% -> "
            f"score={score} ({verdict}); activity category: {act['category']}.")
        ranked.append(RankedDay(
            day=day, score=score, verdict=verdict,
            activities=act["activities"], warnings=warnings))

    ranked_sorted = sorted(ranked, key=lambda r: r.score, reverse=True)
    best = ranked_sorted[0]
    trace_lines.append(
        f"[decide] Ranked {len(ranked)} days. Best: {best.day.day_name} "
        f"({best.score}). Producing final plan.")

    plan_md = _render_plan_markdown(forecast, ranked_sorted, preferences,
                                    engine="rule-based controller")
    return PlanResult(
        city=forecast.city,
        mode=config.mode_label(),
        plan_markdown=plan_md,
        forecast=forecast,
        ranked_days=ranked,            # keep chronological order for the table
        reasoning_trace="\n".join(trace_lines),
        tool_calls=tool_calls,
    )


def _render_plan_markdown(forecast: Forecast, ranked_sorted: list,
                          preferences: str, engine: str) -> str:
    lines = [f"## Travel plan for {forecast.city}", ""]
    if preferences.strip():
        lines.append(f"*Preferences considered:* {preferences.strip()}")
        lines.append("")

    lines.append("### Recommended days (best first)")
    for i, r in enumerate(ranked_sorted, 1):
        d = r.day
        lines.append(
            f"**{i}. {d.day_name} ({d.date}) - {r.verdict} ({r.score}/100)**  ")
        lines.append(
            f"{d.description.capitalize()}, {d.temp_min:.0f}-{d.temp_max:.0f}C, "
            f"rain {int(d.precip_prob*100)}%, wind {d.wind_kph:.0f} km/h.  ")
        lines.append(f"Suggested: {', '.join(r.activities)}.  ")
        for w in r.warnings:
            lines.append(f"> ⚠️ {w}  ")
        lines.append("")

    warn_days = [r for r in ranked_sorted if r.warnings]
    if warn_days:
        lines.append("### Weather warnings")
        for r in warn_days:
            for w in r.warnings:
                lines.append(f"- **{r.day.day_name}:** {w}")
        lines.append("")

    best = ranked_sorted[0]
    poor = [r for r in ranked_sorted if r.verdict == "Poor"]
    summary = (f"The best day to visit {forecast.city} is "
               f"**{best.day.day_name} ({best.day.date})** with a suitability "
               f"score of {best.score}/100.")
    if poor:
        names = ", ".join(r.day.day_name for r in poor)
        summary += f" Consider avoiding outdoor plans on: {names}."
    lines.append("### Summary")
    lines.append(summary)
    lines.append("")
    lines.append(f"*Plan generated by the {engine}.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FULL MODE engine: LangChain tool-calling agent (Groq / LLaMA 3)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an autonomous Travel-Planning Agent.

Your job: given a destination city and trip context, recommend the BEST days to \
visit based on the weather forecast, and suggest weather-appropriate activities.

You MUST use your tools to gather information before answering:
- Call `get_weather` once to retrieve the multi-day forecast for the city.
- For days you are recommending, call `recommend_activities` with that day's \
condition and average temperature to get suitable activity ideas.

Reason step by step:
1. Retrieve the forecast.
2. For each day judge travel suitability using temperature, precipitation \
probability, rain amount, and wind. Penalise thunderstorms, heavy rain, very \
high wind, and temperature extremes.
3. Rank the days from best to worst.
4. Issue clear warnings for any dangerous or unpleasant conditions.
5. Respect the user's stated preferences when choosing activities.

Produce a final answer in Markdown with: a ranked list of recommended days \
(each with date, a short reason, and 2-4 activities), a 'Weather warnings' \
section, and a one-paragraph summary naming the single best day. Be concise and \
practical."""


def _plan_with_llm(forecast: Forecast, city: str, preferences: str,
                   days: int) -> PlanResult:
    """Run the real LangChain tool-calling agent. Imports are local so the rest
    of the app works even when LangChain is not installed."""
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq

    @tool
    def get_weather(city: str) -> str:
        """Get the multi-day weather forecast for a city. Returns one line per
        day with the date, weather condition, temperature range (Celsius),
        precipitation probability, expected rain (mm), wind (km/h) and humidity."""
        return get_weather_forecast(city, max_days=days).as_text()

    @tool
    def recommend_activities(condition: str, temp_avg: float,
                             preferences: str = "") -> str:
        """Recommend activities for a given weather condition (e.g. Clear,
        Clouds, Rain, Thunderstorm) and average temperature in Celsius,
        optionally biased by the user's preferences."""
        res = suggest_activities(condition, temp_avg, preferences)
        return (f"Category: {res['category']}. {res['rationale']} "
                f"Suggested: {', '.join(res['activities'])}.")

    tools = [get_weather, recommend_activities]
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    llm = ChatGroq(model=config.GROQ_MODEL, temperature=0.3,
                   api_key=config.GROQ_API_KEY)
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False,
        return_intermediate_steps=True, max_iterations=8,
        handle_parsing_errors=True,
    )

    user_request = (
        f"Plan a trip to {city}. The trip window is the next {days} days "
        f"(today is {dt.date.today().isoformat()}). "
        f"User preferences: {preferences or 'none specified'}."
    )
    result = executor.invoke({"input": user_request})

    # Build the transparency trace from the agent's actual tool calls.
    trace_lines = [f"[agent] {config.GROQ_MODEL} via Groq - autonomous loop"]
    tool_calls: list[str] = []
    for action, observation in result.get("intermediate_steps", []):
        call = f"{action.tool}({action.tool_input})"
        tool_calls.append(call)
        obs = str(observation)
        trace_lines.append(f"[tool] {call}")
        trace_lines.append(f"[observe] {obs[:300]}{'...' if len(obs) > 300 else ''}")
    trace_lines.append("[decide] Agent synthesised the final plan from observations.")

    # Compute ranked days with the deterministic scorer too, so the UI table and
    # the structured outputs are always populated regardless of LLM formatting.
    ranked = []
    for day in forecast.days:
        score, verdict, warnings = score_day(day)
        act = suggest_activities(day.condition, day.temp_avg, preferences)
        ranked.append(RankedDay(day=day, score=score, verdict=verdict,
                                activities=act["activities"], warnings=warnings))

    return PlanResult(
        city=forecast.city,
        mode=config.mode_label(),
        plan_markdown=result.get("output", "").strip() or
        _render_plan_markdown(forecast, sorted(ranked, key=lambda r: r.score,
                                               reverse=True), preferences,
                              engine="LLM agent"),
        forecast=forecast,
        ranked_days=ranked,
        reasoning_trace="\n".join(trace_lines),
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def plan_trip(city: str, days: int | None = None,
              preferences: str = "") -> PlanResult:
    """Plan a trip to `city`. Chooses the LLM agent when a Groq key is present,
    otherwise the deterministic rule-based controller. Never raises for missing
    keys; on an unexpected LLM error it degrades to the rule-based engine."""
    days = days or config.DEFAULT_TRIP_DAYS
    days = max(1, min(days, config.MAX_FORECAST_DAYS))
    forecast = get_weather_forecast(city, max_days=days)

    if config.USE_LLM:
        try:
            return _plan_with_llm(forecast, city, preferences, days)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            fallback = _plan_rule_based(forecast, preferences)
            fallback.reasoning_trace = (
                f"[warning] LLM agent failed ({type(exc).__name__}: {exc}); "
                f"fell back to the rule-based controller.\n"
                + fallback.reasoning_trace
            )
            return fallback

    return _plan_rule_based(forecast, preferences)
