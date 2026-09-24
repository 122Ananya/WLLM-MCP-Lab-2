"""
The "MCP server" side of this connector: a single tool the LLM can call to
reach outside its own knowledge and pull live data from a real API
(wttr.in — no API key required).
"""

import requests

WTTR_BASE = "https://wttr.in"

# Tool definition in Groq/OpenAI function-calling schema. This is the
# "contract" the LLM sees: name, description, and expected input shape.
GET_CURRENT_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current real-time weather conditions for a city or "
            "location. Returns temperature, feels-like temperature, "
            "conditions, humidity, and wind speed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo', 'Bengaluru', 'New York'",
                }
            },
            "required": ["location"],
        },
    },
}


def get_current_weather(location: str) -> dict:
    """Fetch current weather for a location from wttr.in and return
    structured data, the way an MCP tool handler would."""
    url = f"{WTTR_BASE}/{location}"
    response = requests.get(
        url,
        params={"format": "j1"},
        headers={"User-Agent": "curl/8.0"},  # wttr.in expects a curl-like UA
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    current = (data.get("current_condition") or [None])[0]
    if not current:
        raise ValueError(f'No weather data returned for "{location}"')

    area = (data.get("nearest_area") or [{}])[0]
    resolved_name = ", ".join(
        filter(
            None,
            [
                (area.get("areaName") or [{}])[0].get("value"),
                (area.get("region") or [{}])[0].get("value"),
                (area.get("country") or [{}])[0].get("value"),
            ],
        )
    )

    return {
        "location": resolved_name or location,
        "temperature_c": float(current["temp_C"]),
        "feels_like_c": float(current["FeelsLikeC"]),
        "condition": (current.get("weatherDesc") or [{}])[0].get("value", "Unknown"),
        "humidity_percent": float(current["humidity"]),
        "wind_kmph": float(current["windspeedKmph"]),
        "wind_direction": current.get("winddir16Point"),
        "observed_at": current.get("localObsDateTime"),
    }
