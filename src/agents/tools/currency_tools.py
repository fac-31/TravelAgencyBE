def detect_local_currency(request: dict = None) -> str:
    """
    Detect the local currency based on request IP geolocation or system locale.

    Args:
        request: Optional request dict with shape {"client": {"host": "<ip>"}}.
                If provided, attempts to detect currency from user's IP location.

    Returns:
        3-letter currency code (e.g., "USD", "EUR", "GBP") or "USD" as fallback.
    """
    try:
        # If request provided, try IP-based geolocation first
        if request:
            from .geoip import get_geoip
            geoip = get_geoip(request)
            if geoip and geoip.get("currency"):
                return geoip["currency"]

        # Try system locale
        import locale
        loc = locale.getdefaultlocale()[0]  # e.g., 'en_US'
        if loc:
            country_code = loc.split("_")[1]
            # Basic mapping of country codes to currency
            country_currency_map = {
                "US": "USD",
                "GB": "GBP",
                "EU": "EUR",
                "JP": "JPY",
                "CN": "CNY",
                "HK": "HKD",
                "CA": "CAD",
                "AU": "AUD",
                # Add more as needed
            }
            if country_code in country_currency_map:
                return country_currency_map[country_code]

    except Exception:
        pass

    # Default fallback
    return "USD"