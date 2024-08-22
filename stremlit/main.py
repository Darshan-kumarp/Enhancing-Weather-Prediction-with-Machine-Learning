import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    return mape

def calculate_accuracy(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    mape_max_temp = calculate_mape(true_values[:, 0], predicted_values[:, 0])
    mape_min_temp = calculate_mape(true_values[:, 1], predicted_values[:, 1])
    
    accuracy_max_temp = 100 - mape_max_temp
    accuracy_min_temp = 100 - mape_min_temp
    
    return mae, accuracy_max_temp, accuracy_min_temp

def load_data(city):
    file_map = {
        "Mysore": "weather_mysore.csv",
        "Bangalore": "weather_bangalore.csv",
        "Belgavi": "weather_belgavi.csv",
        "Chikmagalur": "weather_chikmagalur.csv",
        "Chickballapur": "weather_chickballapur.csv"  # Added Chickballapur file
    }
    file_path = file_map.get(city)
    if file_path:
        data = pd.read_csv(file_path)
        return data
    return None

def main():
    st.title("Weather Forecasting")

    # City selection dropdown with "Select City" as default
    city = st.selectbox("Select City", ["Select City", "Mysore", "Bangalore", "Belgavi", "Chikmagalur", "Chickballapur"])  # Added Chickballapur

    if city != "Select City":
        # Load the dataset for the selected city
        data = load_data(city)

        if data is not None:
            # Parse the date column
            data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

            # Data Cleaning
            data.replace(-999, np.nan, inplace=True)
            data.dropna(inplace=True)

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
            mae, accuracy_max_temp, accuracy_min_temp = calculate_accuracy(y_test.values[:, :2], predictions[:, :2])
            st.write("**Random Forest**")
            st.write("Mean Absolute Error:", mae)
            st.write("Max Temp Accuracy:", accuracy_max_temp, "%")
            st.write("Min Temp Accuracy:", accuracy_min_temp, "%")

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
                if precipitation < 1:
                    weather_condition = "Sunny"
                elif 1.01 <= precipitation <= 2.5:
                    weather_condition = "Cloudy"
                elif 2.51 <= precipitation <= 5:
                    weather_condition = "Light Rain"
                elif 5.01 <= precipitation <= 8:
                    weather_condition = "moderate Rain"
                else:
                    weather_condition = "Heavy Rain"

                # Append temperature values with units
                max_temp_str = f"{int(prediction[0])} °C"
                min_temp_str = f"{int(prediction[1])} °C"
                average_temp_str = f"{average_temp} °C"

                # Format precipitation with up to 4 decimal places
                precipitation_str = f"{precipitation:.4f}"

                forecasts.append((max_temp_str, min_temp_str, precipitation_str, average_temp_str, weather_condition))
                last_known['prev_max_temp'], last_known['prev_min_temp'], last_known['prev_precipitation'] = prediction

            forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame(forecasts, columns=['MaxTemp', 'MinTemp', 'PrecipitationAmount', 'AverageTemp', 'Weather'], index=forecast_dates)

            # Increase the width of the Weather column
            styled_df = forecast_df.style.set_properties(
                **{'text-align': 'center'},
                subset=pd.IndexSlice[:, :]
            ).set_table_styles(
                [{'selector': 'th.col5', 'props': [('min-width', '150px')]}]
            )

            st.dataframe(styled_df)

if __name__ == "__main__":
    main()
