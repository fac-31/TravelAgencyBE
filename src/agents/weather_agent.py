"""
Weather agent - provides weather information for travel planning
"""

import requests
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import re
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage


def fetch_weather_data(city: str, date: str) -> str:
    """
    Fetch actual weather forecast data from Open-Meteo API for a given city and date.
    The 'date' can be an exact date (YYYY-MM-DD) or natural language like:
    'today', 'tomorrow', 'next Friday', 'in 3 days', etc.

    This is a data-fetching utility (no LLM involved).

    Example:
        fetch_weather_data("Paris", "tomorrow")
        fetch_weather_data("London", "2025-11-02")
    """
    try:
        # --- Step 1: Parse natural language date ---
        today = datetime.utcnow().date()
        date = date.strip().lower()

        # Handle simple keywords
        if date in ["today"]:
            target_date = today
        elif date in ["tomorrow"]:
            target_date = today + timedelta(days=1)
        elif match := re.match(r"in (\d+) days?", date):
            target_date = today + timedelta(days=int(match.group(1)))
        else:
            # Try using dateutil.parser for "next Friday", etc.
            try:
                target_date = date_parser.parse(date, fuzzy=True).date()
            except Exception:
                return "Please provide a valid date or phrase like 'tomorrow' or 'next Monday'."

        # --- Step 2: Get coordinates for the city ---
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_res = requests.get(geo_url, timeout=5)
        geo_data = geo_res.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find location for city '{city}'."

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        # --- Step 3: Fetch forecast for that date ---
        date_str = target_date.strftime("%Y-%m-%d")
        forecast_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,"
            f"precipitation_probability_mean&timezone=auto&start_date={date_str}&end_date={date_str}"
        )

        forecast_res = requests.get(forecast_url, timeout=5)
        forecast_data = forecast_res.json()
        daily = forecast_data.get("daily", {})

        if not daily or not daily.get("temperature_2m_max"):
            return f"No forecast data available for {city} on {date_str}."

        max_temp = daily["temperature_2m_max"][0]
        min_temp = daily["temperature_2m_min"][0]
        rain_prob = daily.get("precipitation_probability_mean", [None])[0]

        # --- Step 4: Construct response ---
        msg = f"Weather forecast for {city.title()} on {date_str}: {min_temp}°C to {max_temp}°C"
        if rain_prob is not None:
            msg += f", with a {rain_prob}% chance of rain."
        return msg

    except Exception as e:
        return f"Could not fetch weather: {str(e)}"


def weather_agent(user_message: str) -> str:
    """
    Process user message about weather and return response.
    Uses LLM to understand the request and extract location/date info.
    """
    model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)

    system_prompt = SystemMessage(
        content=(
            "You are a weather assistant. Based on the user's request, extract the city name and date "
            "they're asking about. Call get_weather with these parameters. "
            "If you cannot determine what they're asking for, ask for clarification."
        )
    )

    # For simplicity, just use LLM to provide weather response
    response = model.invoke([
        system_prompt,
        HumanMessage(content=user_message)
    ])

    return response.content
