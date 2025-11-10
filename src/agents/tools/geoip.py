"""
GeoIP utility for detecting user location based on IP address.
Results are cached in-memory and persisted to disk for performance.
"""

import requests
import time
import json
import os
from typing import Optional, Dict, Any, Union


# In-memory cache for geoip results keyed by IP address
# Stores (timestamp_seconds, response_dict). TTL defaults to 24 hours.
_GEOIP_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_GEOIP_TTL_SECONDS = 60 * 60 * 24

# File used to persist the cache across process restarts
_CACHE_FILENAME = os.path.join(os.path.dirname(__file__), ".geoip_cache.json")


def _load_cache_from_disk() -> None:
    """Load cache from disk into _GEOIP_CACHE if file exists and is readable."""
    try:
        if not os.path.exists(_CACHE_FILENAME):
            return
        with open(_CACHE_FILENAME, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        now = time.time()
        # raw is expected to be mapping ip -> {"ts": float, "data": dict}
        for key, val in raw.items():
            ts = float(val.get("ts", 0))
            data = val.get("data")
            if data is None:
                continue
            # Only load entries that are not expired relative to TTL
            if now - ts < _GEOIP_TTL_SECONDS:
                _GEOIP_CACHE[key] = (ts, data)
    except Exception as e:
        # Failure to load cache should not stop the app
        print("Failed to load geoip cache from disk:", e)


def _save_cache_to_disk() -> None:
    """Persist _GEOIP_CACHE to disk in a small JSON structure."""
    try:
        to_write: Dict[str, Dict[str, Any]] = {}
        for key, (ts, data) in _GEOIP_CACHE.items():
            to_write[key] = {"ts": ts, "data": data}
        with open(_CACHE_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(to_write, fh)
    except Exception as e:
        print("Failed to save geoip cache to disk:", e)


def _extract_ip(request: Union[dict, None]) -> Optional[str]:
    """Extract IP address from request dict."""
    if request is None:
        return None
    if isinstance(request, dict):
        return request.get("client", {}).get("host")
    return None


def get_geoip(request: Union[dict, None], ttl: int = _GEOIP_TTL_SECONDS) -> Optional[Dict[str, Any]]:
    """
    Get geolocation information for a given IP address.

    Accepts a simplified dict with shape {"client": {"host": "<ip>"}}.
    Results are cached in-memory keyed by IP for `ttl` seconds (default 24h).
    Cache is persisted to `.geoip_cache.json` next to this module so lookups survive reloads.

    Args:
        request: Dict with request info, should contain client IP
        ttl: Cache time-to-live in seconds (default 24h)

    Returns:
        Dict with geolocation data (city, country_name, currency, etc.) or None on error
    """
    try:
        ip = _extract_ip(request)

        # Treat local/loopback as a generic lookup
        key = ip if ip and ip != "127.0.0.1" else "local"

        # Check cache
        entry = _GEOIP_CACHE.get(key)
        now = time.time()
        if entry:
            ts, data = entry
            if now - ts < ttl:
                return data

        # Build lookup URL
        if key == "local":
            url = "https://ipapi.co/json/"
        else:
            url = f"https://ipapi.co/{ip}/json/"

        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        if data.get("error"):
            # Don't cache error results
            print("Error in IP geolocation response:", data.get("reason"), data.get("message"))
            return None

        # Cache the successful response and persist
        _GEOIP_CACHE[key] = (now, data)
        _save_cache_to_disk()
        return data

    except requests.exceptions.RequestException as e:
        print("Error detecting geoip (request):", e)
        return None
    except Exception as e:
        print("Error detecting geoip:", e)
        return None


# Load cached values at import time so they survive reloads
_load_cache_from_disk()
