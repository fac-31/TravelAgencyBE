"""
Flight agent - searches for flight offers using Amadeus API
"""

from fastapi import Request
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

from src.agents.tools.geoip import get_geoip


load_dotenv()


def fetch_flight_offers(origin: str,
                       destination: str,
                       departure_date: str,
                       adults: int = 1,
                       return_date: str | None = None,
                       max_results: int = 5) -> str:
    """
    Fetch flight offers from the Amadeus test API.

    Args:
        origin: 3-letter IATA code (e.g., 'NYC')
        destination: 3-letter IATA code (e.g., 'LAX')
        departure_date: Date in YYYY-MM-DD format
        adults: Number of adult passengers (default 1)
        return_date: Optional return date for round trips
        max_results: Maximum number of results to return (default 5)

    Returns:
        str: Human-friendly summary of flight offers or error message
    """
    client_id = os.getenv("AMDERUS_API_KEY")
    client_secret = os.getenv("AMADEUS_API_SECRET")

    if not client_id or not client_secret:
        return "Error: Amadeus credentials not configured. Please set AMDERUS_API_KEY and AMADEUS_API_SECRET in environment."

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
            return "Error: Could not authenticate with flight service."

        access_token = token_res.json().get("access_token")
        if not access_token:
            return "Error: Could not obtain access token from flight service."

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
        except requests.exceptions.HTTPError:
            return f"Error: Could not search flights from {origin} to {destination}. Check airport codes and dates."

        data = res.json()

        if "data" not in data or not data["data"]:
            return f"No flights found from {origin} to {destination} on {departure_date}."

        # Build readable summary
        lines = []
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

    except requests.exceptions.RequestException:
        return "Error: Could not connect to flight service. Please try again later."
    except Exception as e:
        return f"Error: Unexpected error while searching flights: {str(e)}"


def flight_agent(user_message: str, request: Request) -> str:
    """
    Process user message about flight searches and return response.
    Uses LLM to understand request and extract flight parameters.
    """
    model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)

    geoip = get_geoip(request)

    system_prompt = SystemMessage(
        content=(
            "You are a flight assistant that helps users find flight offers. "
            "Understand the user's request to extract origin, destination, and travel dates. "
            f"Assume the user's home location is based on their geolocation: {geoip.get('city', '')}, {geoip.get('country_name', '')} . "
            "Assume 1 adult passenger and one-way trips unless stated otherwise. "
            "Use 3-letter IATA airport codes (e.g., NYC, LAX, LHR). "
            "Provide helpful responses about flight options."
        )
    )

    response = model.invoke([
        system_prompt,
        HumanMessage(content=user_message)
    ])

    return response.content
