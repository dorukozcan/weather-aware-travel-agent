"""
Tools available to the travel-planning agent.

Two capabilities are exposed:

  1. get_weather_forecast(city)   -> daily weather summaries for the next days
  2. suggest_activities(condition, temp_avg, preferences) -> activity ideas

Both are plain Python functions with clean, typed signatures. In FULL MODE they
are wrapped as LangChain tools (see agent.py) so the LLM can call them
autonomously; in DEMO MODE the rule-based planner calls them directly. Either
way, the SAME tool code runs, which keeps behaviour consistent across modes.

When OPENWEATHER_API_KEY is set the real OpenWeatherMap "5 day / 3 hour" forecast
API is used and aggregated into per-day summaries. Otherwise a deterministic
mock provider produces realistic, stable synthetic forecasts so the system is
fully runnable with no keys.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests

import config


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class DayForecast:
    """Aggregated weather summary for a single day."""

    date: str            # ISO YYYY-MM-DD
    day_name: str        # e.g. "Monday"
    condition: str       # normalised: Clear / Clouds / Rain / Snow / Thunderstorm / Drizzle / Mist
    description: str     # free-text, e.g. "light rain"
    temp_min: float      # Celsius
    temp_max: float      # Celsius
    temp_avg: float      # Celsius
    precip_prob: float   # 0..1 probability of precipitation
    rain_mm: float       # expected rain volume (mm)
    wind_kph: float      # wind speed (km/h)
    humidity: int        # %

    def to_dict(self) -> dict:
        return asdict(self)

    def one_line(self) -> str:
        return (
            f"{self.day_name} {self.date}: {self.description}, "
            f"{self.temp_min:.0f}-{self.temp_max:.0f}C, "
            f"rain {int(self.precip_prob * 100)}% ({self.rain_mm:.1f}mm), "
            f"wind {self.wind_kph:.0f} km/h, humidity {self.humidity}%"
        )


@dataclass
class Forecast:
    city: str
    source: str                       # "openweathermap" or "mock"
    days: list = field(default_factory=list)  # list[DayForecast]

    def to_dict(self) -> dict:
        return {"city": self.city, "source": self.source,
                "days": [d.to_dict() for d in self.days]}

    def as_text(self) -> str:
        head = f"Weather forecast for {self.city} (source: {self.source}):"
        return head + "\n" + "\n".join(d.one_line() for d in self.days)


# Severity ranking used when several conditions occur in the same day: we report
# the most "trip-relevant" (severe) one.
_SEVERITY = {
    "Thunderstorm": 6, "Snow": 5, "Rain": 4, "Drizzle": 3,
    "Mist": 2, "Clouds": 1, "Clear": 0,
}


# ---------------------------------------------------------------------------
# Real provider: OpenWeatherMap
# ---------------------------------------------------------------------------
def _fetch_openweather(city: str, max_days: int) -> Forecast:
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": config.OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()

    resolved_city = payload.get("city", {}).get("name", city)
    country = payload.get("city", {}).get("country", "")
    if country:
        resolved_city = f"{resolved_city}, {country}"

    # Group the 3-hourly entries by calendar date.
    buckets: dict[str, list[dict]] = {}
    for item in payload.get("list", []):
        date_key = item["dt_txt"].split(" ")[0]
        buckets.setdefault(date_key, []).append(item)

    days: list[DayForecast] = []
    for date_key in sorted(buckets)[:max_days]:
        entries = buckets[date_key]
        temps = [e["main"]["temp"] for e in entries]
        temp_mins = [e["main"].get("temp_min", e["main"]["temp"]) for e in entries]
        temp_maxs = [e["main"].get("temp_max", e["main"]["temp"]) for e in entries]
        humidity = [e["main"]["humidity"] for e in entries]
        winds = [e.get("wind", {}).get("speed", 0.0) for e in entries]  # m/s
        pops = [e.get("pop", 0.0) for e in entries]
        rain = sum(e.get("rain", {}).get("3h", 0.0) for e in entries)

        # Choose the most severe condition of the day.
        worst_main = "Clear"
        worst_desc = "clear sky"
        for e in entries:
            main = e["weather"][0]["main"]
            if _SEVERITY.get(main, 0) >= _SEVERITY.get(worst_main, 0):
                worst_main = main
                worst_desc = e["weather"][0]["description"]

        date_obj = dt.date.fromisoformat(date_key)
        days.append(DayForecast(
            date=date_key,
            day_name=date_obj.strftime("%A"),
            condition=worst_main,
            description=worst_desc,
            temp_min=round(min(temp_mins), 1),
            temp_max=round(max(temp_maxs), 1),
            temp_avg=round(sum(temps) / len(temps), 1),
            precip_prob=round(max(pops), 2),
            rain_mm=round(rain, 1),
            wind_kph=round(max(winds) * 3.6, 1),
            humidity=int(round(sum(humidity) / len(humidity))),
        ))

    return Forecast(city=resolved_city, source="openweathermap", days=days)


# ---------------------------------------------------------------------------
# Mock provider: deterministic, realistic synthetic forecasts
# ---------------------------------------------------------------------------
_CONDITION_POOL = [
    ("Clear", "clear sky"),
    ("Clouds", "scattered clouds"),
    ("Clouds", "overcast clouds"),
    ("Rain", "light rain"),
    ("Rain", "moderate rain"),
    ("Drizzle", "light drizzle"),
    ("Thunderstorm", "thunderstorm"),
]


def _mock_forecast(city: str, max_days: int) -> Forecast:
    """Stable per (city, day) synthetic forecast. Same inputs -> same output."""
    today = dt.date.today()
    seed = int(hashlib.sha256(f"{city.lower()}|{today.isoformat()}".encode()).hexdigest(), 16)
    rng = random.Random(seed)

    # A base temperature loosely tied to the city string, so different cities
    # feel different but each city is stable for the day.
    base_temp = 6 + (seed % 26)            # 6..31 C baseline
    days: list[DayForecast] = []
    for i in range(max_days):
        date_obj = today + dt.timedelta(days=i)
        cond, desc = rng.choices(
            _CONDITION_POOL,
            weights=[28, 22, 14, 14, 8, 8, 6],  # mostly fair, occasionally bad
            k=1,
        )[0]

        swing = rng.uniform(3, 8)
        temp_avg = base_temp + rng.uniform(-4, 4) - (2 if cond in ("Rain", "Thunderstorm") else 0)
        temp_min = temp_avg - swing / 2
        temp_max = temp_avg + swing / 2

        if cond == "Thunderstorm":
            precip_prob, rain_mm, wind = rng.uniform(0.7, 0.95), rng.uniform(6, 18), rng.uniform(25, 55)
        elif cond == "Rain":
            precip_prob, rain_mm, wind = rng.uniform(0.5, 0.85), rng.uniform(2, 9), rng.uniform(10, 30)
        elif cond == "Drizzle":
            precip_prob, rain_mm, wind = rng.uniform(0.35, 0.6), rng.uniform(0.3, 2), rng.uniform(8, 20)
        elif cond == "Clouds":
            precip_prob, rain_mm, wind = rng.uniform(0.05, 0.3), rng.uniform(0, 0.4), rng.uniform(6, 22)
        else:  # Clear
            precip_prob, rain_mm, wind = rng.uniform(0.0, 0.1), 0.0, rng.uniform(4, 16)

        days.append(DayForecast(
            date=date_obj.isoformat(),
            day_name=date_obj.strftime("%A"),
            condition=cond,
            description=desc,
            temp_min=round(temp_min, 1),
            temp_max=round(temp_max, 1),
            temp_avg=round(temp_avg, 1),
            precip_prob=round(precip_prob, 2),
            rain_mm=round(rain_mm, 1),
            wind_kph=round(wind, 1),
            humidity=int(rng.uniform(45, 92)),
        ))

    return Forecast(city=city.title(), source="mock", days=days)


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------
def get_weather_forecast(city: str, max_days: Optional[int] = None) -> Forecast:
    """TOOL: return per-day weather summaries for `city`.

    Uses the live OpenWeatherMap API when a key is configured, otherwise a
    deterministic mock provider. Never raises on a missing key; on a network or
    API error it transparently falls back to the mock provider so the agent can
    still produce a plan.
    """
    max_days = max_days or config.MAX_FORECAST_DAYS
    city = (city or "").strip()
    if not city:
        raise ValueError("city must not be empty")

    if config.USE_REAL_WEATHER:
        try:
            fc = _fetch_openweather(city, max_days)
            if fc.days:
                return fc
        except Exception:
            # Fall through to mock so a key/quota/network problem never crashes
            # the user-facing app.
            pass
    return _mock_forecast(city, max_days)


# Activity catalogue keyed by a coarse weather "bucket".
_ACTIVITY_CATALOGUE = {
    "good_outdoor": [
        "city walking tour / sightseeing", "visit parks and gardens",
        "open-air markets", "boat trip or waterfront walk",
        "cycling tour", "rooftop viewpoints and terraces",
    ],
    "mild_mixed": [
        "museum and gallery hopping", "historic neighbourhood walk",
        "cafe and food tour", "short hikes with a rain jacket",
        "shopping districts",
    ],
    "indoor": [
        "museums and art galleries", "indoor markets and malls",
        "thermal baths / spa", "aquarium or science centre",
        "local cuisine restaurant crawl", "cinema or theatre",
    ],
    "hot": [
        "early-morning sightseeing then midday indoor breaks",
        "swimming / water activities", "shaded park picnics",
        "air-conditioned museums in the afternoon",
    ],
    "cold": [
        "warm cafes and bakeries", "indoor cultural sites",
        "winter markets", "museums and galleries",
    ],
}


def suggest_activities(condition: str, temp_avg: float,
                       preferences: str = "") -> dict:
    """TOOL: recommend activities given a weather `condition` and `temp_avg`.

    Returns {category, activities, rationale}. `preferences` is free text used
    only to lightly bias the wording; the core logic is weather-driven.
    """
    condition = (condition or "").title()
    bad = condition in ("Rain", "Thunderstorm", "Snow", "Drizzle")

    if bad:
        category = "indoor"
        rationale = f"{condition} expected, so indoor-friendly activities are safer."
        pool = list(_ACTIVITY_CATALOGUE["indoor"])
    elif temp_avg >= 30:
        category = "hot-weather (mostly outdoor with midday shade)"
        rationale = "High temperatures: plan outdoor time for mornings/evenings."
        pool = list(_ACTIVITY_CATALOGUE["hot"])
    elif temp_avg <= 5:
        category = "cold-weather (mixed)"
        rationale = "Low temperatures: keep mostly to warm indoor options."
        pool = list(_ACTIVITY_CATALOGUE["cold"])
    elif condition in ("Clear",) and 12 <= temp_avg < 30:
        category = "outdoor"
        rationale = "Clear and comfortable: great for outdoor sightseeing."
        pool = list(_ACTIVITY_CATALOGUE["good_outdoor"])
    else:
        category = "mixed (indoor + outdoor)"
        rationale = "Mild/cloudy conditions suit a flexible indoor-outdoor mix."
        pool = list(_ACTIVITY_CATALOGUE["mild_mixed"])

    prefs = (preferences or "").lower()
    if "museum" in prefs or "art" in prefs:
        pool = ["museums and art galleries"] + [a for a in pool if "museum" not in a]
    if "food" in prefs or "cuisine" in prefs:
        pool = ["local cuisine / food tour"] + pool

    return {"category": category, "activities": pool[:4], "rationale": rationale}
