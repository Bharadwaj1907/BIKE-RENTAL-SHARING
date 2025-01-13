import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('bike_rent (1) (1).csv')

# Title and description
st.title("Bike Rental Sharing Prediction")
st.write("This app predicts the number of bikes rented based on user-provided details.")

# Display dataset overview
if st.checkbox("Show Dataset"):
    st.write("### Dataset Overview")
    st.dataframe(data.head())
    st.write(f"### Dataset Shape: {data.shape}")

# Preprocess the data
data['dteday'] = pd.to_datetime(data['dteday'])
data = pd.get_dummies(data, columns=['season', 'weathersit', 'holiday', 'workingday', 'weekday'], drop_first=True)

# Select features and target
features = [
    'yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed',
    'season_summer', 'season_winter', 'season_fall',
    'weathersit_Cloudy', 'weathersit_Light Snow/Rain',
    'holiday_Yes', 'workingday_Yes'
]
X = data[features]
y = data['cnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
st.write("### Model Performance")
st.write(f"- RMSE: {rmse:.2f}")
st.write(f"- R² Score: {r2:.2f}")

# User input for prediction
st.write("## Make a Prediction")

yr = st.selectbox("Year", [2011, 2012])
mnth = st.slider("Month", 1, 12)
hr = st.slider("Hour", 0, 23)
temp = st.number_input("Temperature (normalized)", value=0.5)
atemp = st.number_input("Feels Like Temperature (normalized)", value=0.5)
hum = st.number_input("Humidity (normalized)", value=0.5)
windspeed = st.number_input("Wind Speed (normalized)", value=0.5)
season = st.selectbox("Season", ["spring", "summer", "fall", "winter"])
weathersit = st.selectbox("Weather Situation", ["Clear", "Cloudy", "Light Snow/Rain"])
holiday = st.selectbox("Is it a holiday?", ["No", "Yes"])
workingday = st.selectbox("Is it a working day?", ["No", "Yes"])

# Encode user input
season_mapping = {"spring": [0, 0, 0], "summer": [1, 0, 0], "fall": [0, 0, 1], "winter": [0, 1, 0]}
weather_mapping = {"Clear": [0, 0], "Cloudy": [1, 0], "Light Snow/Rain": [0, 1]}
holiday_mapping = {"No": 0, "Yes": 1}
workingday_mapping = {"No": 0, "Yes": 1}

season_encoded = season_mapping[season]
weather_encoded = weather_mapping[weathersit]

# Combine all inputs
user_input = [
    yr, mnth, hr, temp, atemp, hum, windspeed,
    *season_encoded, *weather_encoded,
    holiday_mapping[holiday], workingday_mapping[workingday]
]

# Predict button
if st.button("Predict Rental Count"):
    prediction = model.predict([user_input])
    st.write(f"### Predicted Rental Count: {prediction[0]:.2f}")
