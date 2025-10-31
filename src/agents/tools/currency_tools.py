def detect_local_currency() -> str:
    """Detect the local currency based on system locale or IP-based geolocation."""
    try:
        # First, try using the system locale
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

        # Fallback: use IP geolocation
        res = requests.get("https://ipapi.co/json/", timeout=5).json()
        currency = res.get("currency")
        if currency:
            print("Currency:", currency)
            return currency

    except Exception:
        pass

    # Default fallback
    return "USD"