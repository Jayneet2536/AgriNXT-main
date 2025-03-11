import requests
import math
import datetime
from flask import Flask, render_template, request
import pickle
import  pandas as pd



app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load your trained model
with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)

# Extract the model and feature names
model = model_data["model"]
feature_names = model_data["feature_names"]

def kelvin_to_celsius(kelvin):
    celsius = kelvin - 273.15
    return celsius

# Function to get weather data
def get_weather_data(city):
    """Fetch data from OpenWeatherMap API."""
    api_key = "3eddcb97ddea03bf887cf0841b51ff0a"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return data


# Function to process the features
def process_features(api_data):
    """Calculate features for the model."""
    # Extract raw data
    pressure = api_data['main']['pressure']
    humidity = api_data['main']['humidity']
    wind_speed = api_data['wind']['speed']
    wind_deg = api_data['wind']['deg']
    cloud_cover = api_data['clouds']['all']
    temp_k = api_data['main']['temp']

    # Calculate dewpoint
    temp_c = temp_k - 273.15
    alpha = (math.log(humidity / 100) + (17.62 * temp_c) / (243.12 + temp_c))
    dewpoint = (243.12 * alpha) / (17.62 - alpha)

    # Calculate sunshine
    sunrise = datetime.datetime.fromtimestamp(api_data['sys']['sunrise'])
    sunset = datetime.datetime.fromtimestamp(api_data['sys']['sunset'])
    daylight_hours = (sunset - sunrise).seconds / 3600
    sunshine = daylight_hours * (1 - cloud_cover / 100)

    # Prepare feature array in SAME ORDER as model training
    features = [
        pressure,
        dewpoint,
        humidity,
        cloud_cover,
        sunshine,
        wind_deg,
        wind_speed
    ]
    return features


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']
        weather_data = get_weather_data(city)
        features = process_features(weather_data)

        # Convert features into a DataFrame before prediction
        input_df = pd.DataFrame([features], columns=feature_names)

        # Call the predict function on the model
        prediction = model.predict(input_df)[0]

        # Interpret the result
        result = "Rainfall" if prediction == 1 else "No Rainfall"

        # Extract weather parameters to pass to the template
        pressure = weather_data['main']['pressure']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']
        cloud_cover = weather_data['clouds']['all']
        # description = weather_data['weather']['description']
        tempreture =  weather_data['main']['temp']
        tempreture_k = int(kelvin_to_celsius(tempreture))
        cloud = weather_data['weather'][0]['main']
        feel_like = weather_data['main']['feels_like']
        feels_like_k = kelvin_to_celsius(feel_like)

        return render_template('index.html', result=result, city=city, pressure=pressure, humidity=humidity, wind_speed=wind_speed, cloud_cover=cloud_cover, feels_like = feels_like_k, clouds = cloud, tempreture = tempreture_k)

    return render_template('index.html', result=None)



if __name__ == "__main__":
    app.run(debug=True, ssl_context=None)
