import requests
from langchain_core.tools import tool


CITY_COORDINATES = {
    "gironde": (44.8378, -0.5792),
    "france": (46.2276, 2.2137),
    "marseille": (43.2965, 5.3698),
    "bordeaux": (44.8378, -0.5792),
    "greece": (39.0742, 21.8243),
    "athens": (37.9838, 23.7275),
    "portugal": (39.3999, -8.2245),
    "lisbon": (38.7223, -9.1393),
    "spain": (40.4168, -3.7038),
    "madrid": (40.4168, -3.7038),
    "turkey": (38.9637, 35.2433),
    "antalya": (36.8969, 30.7133),
    "italy": (41.8719, 12.5674),
    "rome": (41.9028, 12.4964),
}


@tool
def get_weather_conditions(location: str) -> str:
    """
    Get current weather conditions for a location to assess wildfire risk.
    Returns temperature, wind speed, humidity and wildfire risk assessment.

    Args:
        location: City or country name (e.g. 'gironde', 'greece', 'marseille')

    Returns:
        Weather conditions and wildfire risk assessment string.
    """
    location_lower = location.lower().strip()

    # Find coordinates
    coords = None
    for key, coord in CITY_COORDINATES.items():
        if key in location_lower or location_lower in key:
            coords = coord
            break

    if not coords:
        return f"Location '{location}' not found. Available: {', '.join(CITY_COORDINATES.keys())}"

    lat, lon = coords

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,"
            f"wind_speed_10m,precipitation"
            f"&timezone=auto"
        )
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data["current"]

        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind = current["wind_speed_10m"]
        precip = current["precipitation"]

        # Wildfire risk assessment
        risk_score = 0
        if temp > 35:
            risk_score += 3
        elif temp > 25:
            risk_score += 1
        if humidity < 20:
            risk_score += 3
        elif humidity < 40:
            risk_score += 1
        if wind > 40:
            risk_score += 3
        elif wind > 20:
            risk_score += 1
        if precip == 0:
            risk_score += 1

        if risk_score >= 7:
            risk_level = "🔴 EXTREME"
        elif risk_score >= 4:
            risk_level = "🟠 HIGH"
        elif risk_score >= 2:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"

        return (
            f"Current weather in {location.title()}:\n"
            f"🌡️ Temperature: {temp}°C\n"
            f"💧 Humidity: {humidity}%\n"
            f"💨 Wind speed: {wind} km/h\n"
            f"🌧️ Precipitation: {precip} mm\n"
            f"🔥 Wildfire risk: {risk_level} (score: {risk_score}/10)"
        )

    except Exception as e:
        return f"Weather data unavailable for {location}: {str(e)}"


if __name__ == "__main__":
    print(get_weather_conditions.invoke("gironde"))
    print(get_weather_conditions.invoke("greece"))