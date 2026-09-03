import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT = 5.0  


class WeatherAPIError(Exception):
    pass


class CityNotFoundError(WeatherAPIError):
    pass


def get_weather(city_name: str) -> dict:
    city_clean = city_name.strip()
    if not city_clean:
        raise CityNotFoundError("Назва міста не може бути порожньою.")

    geo_params = {
        "name": city_clean,
        "count": 1,
        "language": "uk",
        "format": "json"
    }

    try:
        geo_response = requests.get(GEOCODING_URL, params=geo_params, timeout=DEFAULT_TIMEOUT)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
    except requests.exceptions.Timeout:
        raise WeatherAPIError("Сервер геокодування не відповів вчасно (Timeout).")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        if status_code and 400 <= status_code < 500:
            raise WeatherAPIError(f"Некоректний запит до сервісу геокодування (Код {status_code}).")
        raise WeatherAPIError(f"Сервер геокодування недоступний (Код {status_code}).")
    except (requests.exceptions.RequestException, ValueError):
        raise WeatherAPIError("Мережева помилка під час звернення до сервісу геокодування.")

    results = geo_data.get("results")
    if not results:
        raise CityNotFoundError(f"Місто '{city_clean}' не знайдено.")

    first_result = results[0]
    latitude = first_result.get("latitude")
    longitude = first_result.get("longitude")
    resolved_name = first_result.get("name", city_clean)

    if latitude is None or longitude is None:
        raise WeatherAPIError("Не вдалося отримати координати для вказаного міста.")

    forecast_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True
    }

    try:
        forecast_response = requests.get(FORECAST_URL, params=forecast_params, timeout=DEFAULT_TIMEOUT)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()
    except requests.exceptions.Timeout:
        raise WeatherAPIError("Сервер погоди не відповів вчасно (Timeout).")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else None
        if status_code and 400 <= status_code < 500:
            raise WeatherAPIError(f"Некоректний запит до сервісу погоди (Код {status_code}).")
        raise WeatherAPIError(f"Сервер погоди тимчасово недоступний (Код {status_code}).")
    except (requests.exceptions.RequestException, ValueError):
        raise WeatherAPIError("Мережева помилка під час отримання даних про погоду.")

    current_weather = forecast_data.get("current_weather")
    if not current_weather or "temperature" not in current_weather or "windspeed" not in current_weather:
        raise WeatherAPIError("Отримано неповні або некоректні дані про погоду.")

    return {
        "city": resolved_name,
        "temperature": current_weather["temperature"],
        "windspeed": current_weather["windspeed"]
    }