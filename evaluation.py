"""
Evaluation framework (the REQUIRED "LLM in the loop" component).

A second model independently judges whether the planning agent succeeded. It
scores each plan on four dimensions (1-5) and returns an overall pass/fail with
written feedback:

  * accuracy    - does the plan faithfully reflect the forecast it was given?
  * usefulness  - are the activity suggestions concrete and relevant?
  * safety      - are warnings issued for genuinely adverse days?
  * coherence   - is the plan well-structured and easy to act on?

In FULL MODE the critic is a Groq / LLaMA 3 model (temperature 0) prompted to
return strict JSON. In DEMO MODE a transparent heuristic evaluator checks the
rendered plan against the forecast. `run_evaluation.py` drives this module over
several scenarios - including deliberately degraded plans - to demonstrate that
the evaluator actually discriminates good plans from bad ones.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import config
from tools import Forecast
from agent import score_day


@dataclass
class EvalResult:
    accuracy: float
    usefulness: float
    safety: float
    coherence: float
    overall: float
    passed: bool
    feedback: str
    evaluator: str            # "llm-critic" or "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


_ACTIVITY_WORDS = [
    "museum", "gallery", "sightseeing", "park", "tour", "market", "walk",
    "spa", "bath", "cuisine", "restaurant", "cinema", "theatre", "aquarium",
    "swimming", "shopping", "hike", "cycling", "boat", "picnic",
]


# ---------------------------------------------------------------------------
# Heuristic evaluator (DEMO MODE) - independent of the planner's internal scores
# ---------------------------------------------------------------------------
def heuristic_evaluate(forecast: Forecast, plan_markdown: str) -> EvalResult:
    text = (plan_markdown or "").lower()
    days = forecast.days
    n = max(1, len(days))

    # Accuracy: how many forecast days are actually referenced in the plan?
    covered = sum(1 for d in days
                  if d.date in text or d.day_name.lower() in text)
    coverage = covered / n
    accuracy = 1 + 4 * coverage

    # Safety: of the days that genuinely need a warning, how many are flagged?
    needed, flagged = 0, 0
    for d in days:
        _, _, warnings = score_day(d)
        if warnings:
            needed += 1
            risky = any(w in text for w in
                        ["warning", "⚠", "storm", "heavy", "wind",
                         "hot", "freez", "avoid"])
            if d.day_name.lower() in text and risky:
                flagged += 1
    safety_recall = 1.0 if needed == 0 else flagged / needed
    safety = 1 + 4 * safety_recall

    # Usefulness: variety of concrete activity suggestions present.
    activity_hits = sum(1 for w in _ACTIVITY_WORDS if w in text)
    usefulness = 1 + 4 * min(1.0, activity_hits / 5.0)

    # Coherence: structural completeness of the rendered plan.
    coherence = 5.0
    if "recommend" not in text:
        coherence -= 1.5
    if "summary" not in text:
        coherence -= 1.5
    if len(text) < 200:
        coherence -= 2.0
    coherence = max(1.0, coherence)

    overall = round((accuracy + usefulness + safety + coherence) / 4, 2)
    passed = overall >= 3.5 and safety_recall >= 0.5 and coverage >= 0.6

    fb = (f"coverage={coverage:.0%}, safety_recall="
          f"{(1.0 if needed == 0 else flagged/needed):.0%} "
          f"({flagged}/{needed} risky days flagged), "
          f"activity_variety={activity_hits} keywords.")
    return EvalResult(
        accuracy=round(accuracy, 2), usefulness=round(usefulness, 2),
        safety=round(safety, 2), coherence=round(coherence, 2),
        overall=overall, passed=passed, feedback=fb, evaluator="heuristic")


# ---------------------------------------------------------------------------
# LLM critic (FULL MODE)
# ---------------------------------------------------------------------------
_CRITIC_PROMPT = """You are a strict evaluation agent (a "critic"). You are given \
the raw weather forecast that a travel-planning agent received, and the travel \
plan it produced. Judge ONLY whether the plan is well-supported by the forecast.

Score each dimension from 1 (poor) to 5 (excellent):
- accuracy: does the plan correctly reflect the forecast (right days ranked \
high, no claims that contradict the data)?
- usefulness: are the recommended activities concrete and appropriate?
- safety: does it warn about adverse days (storms, heavy rain, extreme heat/cold, \
high wind) present in the forecast?
- coherence: is the plan clear, structured, and actionable?

Return ONLY a JSON object, no prose, with keys:
{{"accuracy": int, "usefulness": int, "safety": int, "coherence": int, \
"passed": bool, "feedback": "one or two sentences"}}

FORECAST:
{forecast}

PLAN:
{plan}
"""


def _parse_json_block(raw: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in critic response")
    return json.loads(match.group(0))


def llm_evaluate(forecast: Forecast, plan_markdown: str) -> EvalResult:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    llm = ChatGroq(model=config.GROQ_MODEL, temperature=0.0,
                   api_key=config.GROQ_API_KEY)
    msg = _CRITIC_PROMPT.format(forecast=forecast.as_text(),
                                plan=plan_markdown)
    response = llm.invoke([HumanMessage(content=msg)])
    data = _parse_json_block(response.content)

    acc = float(data.get("accuracy", 3))
    use = float(data.get("usefulness", 3))
    saf = float(data.get("safety", 3))
    coh = float(data.get("coherence", 3))
    overall = round((acc + use + saf + coh) / 4, 2)
    passed = bool(data.get("passed", overall >= 3.5))
    return EvalResult(
        accuracy=acc, usefulness=use, safety=saf, coherence=coh,
        overall=overall, passed=passed,
        feedback=str(data.get("feedback", "")), evaluator="llm-critic")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def evaluate_plan(forecast: Forecast, plan_markdown: str) -> EvalResult:
    """Evaluate a plan with the LLM critic when available, else the heuristic
    evaluator. Falls back to heuristic on any LLM error."""
    if config.USE_LLM:
        try:
            return llm_evaluate(forecast, plan_markdown)
        except Exception:
            pass
    return heuristic_evaluate(forecast, plan_markdown)
