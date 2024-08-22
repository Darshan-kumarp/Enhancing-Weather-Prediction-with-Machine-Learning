import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def calculate_accuracy(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    return mae

def load_data(city):
    file_map = {
        "Mysore": "weather_mysore.csv",
        "Bangalore": "weather_bangalore.csv",
        "Belgavi": "weather_belgavi.csv"
    }
    file_path = file_map.get(city, None)
    if file_path is None:
        return None
    data = pd.read_csv(file_path)
    return data

def main():
    st.title("Weather Forecasting")

    # City selection dropdown
    city_options = ["Select your city", "Mysore", "Bangalore", "Belgavi"]
    city = st.selectbox("Select City", city_options)

    if city == "Select your city":
        st.info("Please select a city.")
        return

    # Load the dataset for the selected city
    data = load_data(city)

    # Parse the date column
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

    # Sort data by date
    data = data.sort_values(by='Date')

    # Feature Engineering
    data['prev_max_temp'] = data['MaxTemp'].shift(1)
    data['prev_min_temp'] = data['MinTemp'].shift(1)
    data['prev_precipitation'] = data['PrecipitationAmount'].shift(1)
    data = data.dropna()

    features = ['prev_max_temp', 'prev_min_temp', 'prev_precipitation']
    target = ['MaxTemp', 'MinTemp', 'PrecipitationAmount']

    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Initialize RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    st.subheader("Model Performance")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = calculate_accuracy(y_test, predictions)
    st.write("*Random Forest*")
    st.write("Mean Absolute Error:", mae)

    st.subheader("Forecast for the next 7 days")

    forecast_days = 7
    last_known = data.iloc[-1].copy()
    forecasts = []

    for _ in range(forecast_days):
        features = np.array([[last_known['prev_max_temp'], last_known['prev_min_temp'], last_known['prev_precipitation']]])
        prediction = model.predict(features)[0]
        
        # Calculate average temperature
        average_temp = int((prediction[0] + prediction[1]) / 2)
        
        # Weather condition logic based on precipitation ranges
        precipitation = prediction[2]
        if precipitation < 2:
            weather_condition = "Sunny"
        elif 2.01 <= precipitation <= 8:
            weather_condition = "Cloudy"
        elif 8.01 <= precipitation <= 12:
            weather_condition = "Light Rain"
        else:
            weather_condition = "Rainy"
        
        # Append temperature values with units
        max_temp_str = f"{int(prediction[0])} °C"
        min_temp_str = f"{int(prediction[1])} °C"
        average_temp_str = f"{average_temp} °C"
        
        # Format precipitation with up to 4 decimal places
        precipitation_str = f"{precipitation:.4f} mm"
        
        forecasts.append((max_temp_str, min_temp_str, precipitation_str, average_temp_str, weather_condition))
        last_known['prev_max_temp'], last_known['prev_min_temp'], last_known['prev_precipitation'] = prediction

    forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame(forecasts, columns=['MaxTemp', 'MinTemp', 'PrecipitationAmount', 'AverageTemp', 'Weather'], index=forecast_dates)

    st.dataframe(forecast_df.style.set_properties({'text-align': 'center'}, subset=pd.IndexSlice[:, :]))

if _name_ == "_main_":
    main()