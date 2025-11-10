import requests
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import re
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage, AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

@tool
def get_weather(city: str, date: str) -> str:
    """
    Get the weather forecast for a given city and date.
    The 'date' can be an exact date (YYYY-MM-DD) or natural language like:
    'today', 'tomorrow', 'next Friday', 'in 3 days', etc.

    Example:
        get_weather("Paris", "tomorrow")
        get_weather("London", "2025-11-02")
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

model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
model_with_tool = model.bind_tools([get_weather])

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def llm_call(state: dict):
    return {
        "messages": [
            model_with_tool.invoke(
                [SystemMessage(content="You are a weather assistant that helps users with forecasts.")] + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    last = state["messages"][-1]
    results = []
    for tool_call in last.tool_calls:
        tool = get_weather
        obs = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))
    # Append tool results to the existing messages list so the tool_result
    # blocks refer to the AI message containing the matching tool_use.
    return {"messages": state["messages"] + results}

def should_continue(state: MessagesState):
    return "tool_node" if state["messages"][-1].tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
graph.add_edge("tool_node", "llm_call")

weather_agent = graph.compile()
