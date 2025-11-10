"""
Exchange rate agent - provides currency conversion information
"""

from fastapi import Request
import requests
import os
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.agents.tools.geoip import get_geoip


def fetch_exchange_rate_data(from_currency: str, to_currency: str) -> str:
    """
    Fetch actual exchange rate data from ExchangeRate-API.
    This is a data-fetching utility (no LLM involved).
    """
    load_dotenv()
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")

    if not api_key:
        return "Error: Exchange rate API key not configured."

    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"

    try:
        res = requests.get(url, timeout=5)
        data = res.json()

        if data.get("result") == "success":
            rate = data["conversion_rate"]
            return f"1 {from_currency} = {rate} {to_currency}"
        else:
            return f"API error: {data.get('error-type', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"


def exchange_agent(user_message: str, request: Request) -> str:
    """
    Process user message about currency exchange and return response.
    Uses LLM to understand the request.
    """
    model = init_chat_model("anthropic:claude-sonnet-4-5", temperature=0)

    geoip = get_geoip(request)

    system_prompt = SystemMessage(
        content=(
            "You are a currency exchange assistant. Help the user with currency conversion questions. "
            f"Assume the user's local currency is based on their geolocation: {geoip.get('currency', 'USD')} . "
            "Provide exchange rate information in a helpful and clear way."
        )
    )

    response = model.invoke([
        system_prompt,
        HumanMessage(content=user_message)
    ])

    return response.content
