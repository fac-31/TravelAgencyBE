import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from .tools.geoip import get_geoip


load_dotenv()


@tool
def get_flights(origin: str,
                destination: str,
                departure_date: str,
                adults: int = 1,
                return_date: str | None = None,
                max_results: int = 5) -> str:
    """Search flights using the Amadeus test API and return a human-friendly summary.
    """
    client_id = os.getenv("AMDERUS_API_KEY")
    client_secret = os.getenv("AMADEUS_API_SECRET")

    if not client_id or not client_secret:
        return "Error: Amadeus client id/secret not found in environment variables (AMA_CLIENT_ID / AMA_CLIENT_SECRET)."

    try:
        # --- Get access token ---
        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        token_res = requests.post(token_url,
                                  headers={"Content-Type": "application/x-www-form-urlencoded"},
                                  data={
                                      "grant_type": "client_credentials",
                                      "client_id": client_id,
                                      "client_secret": client_secret,
                                  },
                                  timeout=8)
        try:
            token_res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = token_res.status_code
            detail = token_res.json() if token_res.text else {}
            error_msg = f"Failed to get access token (HTTP {status}): {detail.get('error_description') or detail.get('error') or str(e)}"
            print("\nAmadeus Auth Error:")
            print("-" * 60)
            print(f"Status: {status}")
            print(f"Response: {token_res.text}")
            print(f"Headers: {dict(token_res.headers)}")
            print("-" * 60)
            return "Sorry, there was a problem authenticating with the flight search service. Please try again later."
            
        access_token = token_res.json().get("access_token")
        if not access_token:
            print("\nAmadeus Auth Error: No access token in response")
            print("-" * 60)
            print(f"Response: {token_res.text}")
            print("-" * 60)
            return "Sorry, there was a problem authenticating with the flight search service. Please try again later."

        # --- Search offers ---
        base_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "max": max_results,
        }
        if return_date:
            params["returnDate"] = return_date

        res = requests.get(base_url, headers=headers, params=params, timeout=10)
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = res.status_code
            detail = res.json() if res.text else {}
            
            # Log detailed error info to console for debugging
            print("\nAmadeus Flight Search Error:")
            print("-" * 60)
            print(f"Status: {status}")
            print(f"Parameters: {params}")
            print(f"Response: {res.text}")
            print(f"Headers: {dict(res.headers)}")
            
            # For validation errors, log them separately
            if status == 400 and 'errors' in detail:
                print("\nValidation Errors:")
                for err in detail['errors']:
                    print(f"- Parameter: {err.get('parameter', 'unknown')}")
                    print(f"  Detail: {err.get('detail', str(err))}")
            print("-" * 60)
            
            # Return a user-friendly message
            if status == 400:
                return f"Sorry, there was a problem with the flight search request for {origin} to {destination}. Please check the dates and airport codes."
            return "Sorry, there was a problem searching for flights. Please try again later."
            
        data = res.json()

        if "data" not in data or not data["data"]:
            return f"No flights found from {origin} to {destination} on {departure_date}."

        # Build a readable summary for the top offers
        lines: list[str] = []
        offers = data["data"][:max_results]
        lines.append(f"Found {len(offers)} offer(s) for {origin}->{destination} on {departure_date}:")

        for idx, offer in enumerate(offers, start=1):
            price = offer.get("price", {}).get("total")
            currency = offer.get("price", {}).get("currency")
            lines.append(f"\nOption {idx}: {currency} {price}")

            for itin_idx, itin in enumerate(offer.get("itineraries", []), start=1):
                duration = itin.get("duration")
                lines.append(f"  Itinerary {itin_idx} (Duration: {duration}):")
                for seg_idx, seg in enumerate(itin.get("segments", []), start=1):
                    carrier = seg.get("carrierCode")
                    number = seg.get("number")
                    dep = seg.get("departure", {})
                    arr = seg.get("arrival", {})
                    dep_code = dep.get("iataCode")
                    arr_code = arr.get("iataCode")
                    dep_at = dep.get("at")
                    arr_at = arr.get("at")
                    lines.append(f"    Segment {seg_idx}: {carrier}{number} {dep_code}->{arr_code} Depart: {dep_at} Arrive: {arr_at}")

        return "\n".join(lines)

    except requests.exceptions.RequestException as e:
        # Log detailed error for debugging
        print("\nAmadeus Request Error:")
        print("-" * 60)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        if hasattr(e, 'request'):
            print(f"Request URL: {e.request.url}")
            print(f"Request Method: {e.request.method}")
            print(f"Request Headers: {dict(e.request.headers)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response Status: {e.response.status_code}")
            print(f"Response Text: {e.response.text}")
        print("-" * 60)
        return "Sorry, there was a problem connecting to the flight service. Please try again later."
    except Exception as e:
        # Log unexpected errors
        print("\nUnexpected Error in Flight Search:")
        print("-" * 60)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("-" * 60)
        return "Sorry, an unexpected error occurred while searching for flights. Please try again later."


model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)
model_with_tool = model.bind_tools([get_flights])


def llm_call(state: dict):
    res = get_geoip(state["request"])

    return {
        "messages": [
            model_with_tool.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a flight assistant that helps users find flight offers using the Amadeus API. "
                            + "Assume its a one-way adult trip on " + datetime.now().strftime("%Y-%m-%d")
                            + (", departing from " + res.get("city", "") + ", " + res.get("country_name", "") + "." if res else ". ")
                            + "Departure and destination must be in 3-letter city code format (e.g., 'NYC' for New York City)."
                        )
                    )
                ] + state["messages"]
            )
        ],
        "request": state["request"],
    }


def tool_node(state: dict):
    last = state["messages"][-1]
    results = []
    for tool_call in last.tool_calls:
        # only one tool is bound here: get_flights
        obs = get_flights.invoke(tool_call["args"])
        results.append(ToolMessage(content=obs, tool_call_id=tool_call["id"]))
    
    # Append tool results to the existing message history so the tool_result
    # blocks reference the AI message that contained the corresponding tool_use.
    return {"messages": state["messages"] + results, "request": state["request"]}


def should_continue(state: dict):
    return "tool_node" if state["messages"][-1].tool_calls else END


graph = StateGraph(dict)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
graph.add_edge("tool_node", "llm_call")

flight_agent = graph.compile()
