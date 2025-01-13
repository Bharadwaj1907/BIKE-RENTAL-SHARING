import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# Load the dataset
data = pd.read_csv('bike_rent (1) (1).csv')

# Title and description
st.title("Bike Rental Sharing Analysis and Prediction")
st.write("This app analyzes bike rental data and predicts rental counts using machine learning models.")

# Display dataset overview
if st.checkbox("Show Dataset"):
    st.write("### Dataset Overview")
    st.dataframe(data.head())
    st.write(f"### Dataset Shape: {data.shape}")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# EDA Section
st.write("## Exploratory Data Analysis")

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    st.write("### Correlation Heatmap")
    correlation_matrix = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# Distribution of target variable 'cnt'
if st.checkbox("Show Target Variable Distribution"):
    st.write("### Distribution of Bike Rentals (cnt)")
    fig, ax = plt.subplots()
    sns.histplot(data['cnt'], kde=True, color='blue', ax=ax)
    ax.set_title('Distribution of Bike Rentals')
    ax.set_xlabel('Rental Count')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# Feature Engineering
data['dteday'] = pd.to_datetime(data['dteday'])
data['year'] = data['dteday'].dt.year
data['month'] = data['dteday'].dt.month
data['day'] = data['dteday'].dt.day
data['weekday'] = data['dteday'].dt.weekday
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Encode categorical variables
data = pd.get_dummies(data, columns=['season', 'weathersit'], drop_first=True)

# Select features and target
X = data[['temp', 'atemp', 'hum', 'windspeed', 'is_weekend']]
y = data['cnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training
model_options = ["Random Forest", "XGBoost"]
model_choice = st.selectbox("Select a Model", model_options)

if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_choice == "XGBoost":
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
st.write(f"### Model Performance ({model_choice})")
st.write(f"- RMSE: {rmse:.2f}")
st.write(f"- R² Score: {r2:.2f}")

# Visualization of predictions
if st.checkbox("Show Predictions vs Actuals"):
    st.write("### Predictions vs Actuals")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_title("Predictions vs Actuals")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    st.pyplot(fig)

# User prediction
st.write("## Make a Prediction")
temp = st.number_input("Enter Temperature (normalized)", value=0.5)
atemp = st.number_input("Enter Feels Like Temperature (normalized)", value=0.5)
hum = st.number_input("Enter Humidity (normalized)", value=0.5)
windspeed = st.number_input("Enter Windspeed (normalized)", value=0.5)
is_weekend = st.selectbox("Is it a weekend?", ["No", "Yes"])
is_weekend = 1 if is_weekend == "Yes" else 0

# Predict button
if st.button("Predict Rental Count"):
    user_input = np.array([[temp, atemp, hum, windspeed, is_weekend]])
    prediction = model.predict(user_input)
    st.write(f"### Predicted Rental Count: {prediction[0]:.2f}")
