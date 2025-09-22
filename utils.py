
# utils.py
import requests


API_KEY="d1a8a1d47c528c72d59e125177cbcdef" # Replace with your API key

def get_weather_data(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"]
        }
    else:
        return None

def get_gps_disease_risk(lat, lon):
    weather = get_weather_data(lat, lon)
    if weather is None:
        return {
            "risk_level": "Unknown",
            "warning": "Unable to fetch weather data.",
            "precautions": "Check your internet or GPS and try again."
        }

    temp = weather["temperature"]
    humidity = weather["humidity"]

    # Simple rules (adjust as needed based on crop/disease type)
    if temp > 30 and humidity > 70:
        risk = "High"
        warning = "High risk of fungal diseases in current conditions."
        precautions = "Apply fungicide. Ensure good airflow. Avoid over-watering."
    elif temp > 25 and humidity > 60:
        risk = "Medium"
        warning = "Moderate risk of disease in this weather."
        precautions = "Monitor crops. Use preventive measures."
    else:
        risk = "Low"
        warning = "Low risk of plant disease in this area."
        precautions = "Normal monitoring is enough."

    return {
        "risk_level": risk,
        "warning": warning,
        "precautions": precautions,
        "temperature": temp,
        "humidity": humidity
    }
