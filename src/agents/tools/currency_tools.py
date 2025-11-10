import requests
from fastapi import Request
from .geoip import get_geoip

def detect_local_currency(request: dict) -> str:
    """Detect local currency from request info dictionary."""
    # Use IP geolocation API to get currency info
    res = get_geoip(request)
    
    if res:
        currency = res.get("currency")
        if currency:
            print("Currency:", currency)
            return currency

    # Default fallback
    return "USD"