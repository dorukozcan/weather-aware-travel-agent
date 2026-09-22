"""
Central configuration for the Weather-Aware Travel Planning Agent.

All secrets are read from environment variables so that NO API key is ever
committed to the repository. The app is designed to run in two modes:

  * DEMO MODE   - no API keys present. A deterministic mock weather provider and
                  a rule-based planner/evaluator are used so the whole pipeline
                  still runs and can be graded offline.
  * FULL MODE   - GROQ_API_KEY (and optionally OPENWEATHER_API_KEY) are present.
                  A real LangChain tool-calling agent powered by Groq / LLaMA 3
                  drives the autonomous loop, and a real LLM critic evaluates it.

Set keys locally with a .env file (see .env.example) or, on Hugging Face Spaces,
through Settings -> Variables and secrets.
"""

from __future__ import annotations

import os

# Optional: load a local .env file if python-dotenv is installed. This is a
# convenience for local development and is a no-op in production if the package
# or file is absent.
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _clean(value: str | None) -> str:
    return (value or "").strip()


# --- Secrets / keys ---------------------------------------------------------
GROQ_API_KEY: str = _clean(os.environ.get("GROQ_API_KEY"))
OPENWEATHER_API_KEY: str = _clean(os.environ.get("OPENWEATHER_API_KEY"))

# Groq model id. Model availability on Groq changes over time; if the default
# is ever deprecated, override it via the GROQ_MODEL environment variable.
# See https://console.groq.com/docs/models for the current list.
GROQ_MODEL: str = _clean(os.environ.get("GROQ_MODEL")) or "llama-3.3-70b-versatile"

# --- Derived capability flags ----------------------------------------------
USE_LLM: bool = bool(GROQ_API_KEY)
USE_REAL_WEATHER: bool = bool(OPENWEATHER_API_KEY)

# --- Planning defaults ------------------------------------------------------
MAX_FORECAST_DAYS: int = 5          # OpenWeatherMap free tier = 5-day forecast
DEFAULT_TRIP_DAYS: int = 5
REQUEST_TIMEOUT: int = 15           # seconds for HTTP calls


def mode_label() -> str:
    """Human-readable description of the current runtime mode."""
    if USE_LLM and USE_REAL_WEATHER:
        return "FULL AGENT MODE - Groq LLM + live OpenWeatherMap data"
    if USE_LLM and not USE_REAL_WEATHER:
        return "AGENT MODE - Groq LLM + mock weather (no OpenWeather key)"
    if not USE_LLM and USE_REAL_WEATHER:
        return "DEMO MODE - rule-based planner + live OpenWeatherMap data"
    return "DEMO MODE - rule-based planner + mock weather (no API keys)"
