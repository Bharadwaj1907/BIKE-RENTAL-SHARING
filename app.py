import streamlit as st
import pandas as pd
import io

# Try to import nbformat and handle the ImportError gracefully
try:
    import nbformat as nbf
    nbformat_installed = True
except ImportError:
    nbformat_installed = False

# Streamlit app configuration
st.set_page_config(page_title="Bike Rental Analysis", page_icon="🚴", layout="centered")

# Custom CSS for background and buttons
st.markdown(
    """
    <style>
    body {
        background-color: #e6f7ff;  /* Light blue background */
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Bike Rental Analysis")

# Load the CSV file
try:
    data = pd.read_csv('bike_rent (1) (1).csv')
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("CSV file not found. Please ensure the file is in the same directory as this app.")
    st.stop()

# Clean data (replace '?' with NaN and convert to numeric where applicable)
data.replace('?', pd.NA, inplace=True)
data = data.apply(pd.to_numeric, errors='ignore')

# Convert `dteday` column to datetime
if 'dteday' in data.columns:
    data['dteday'] = pd.to_datetime(data['dteday'], errors='coerce')

# Center-align all selection options
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

# Individual feature selection
st.write("### Select Features")
columns = data.columns.tolist()
selected_values = {}

for col in columns:
    if col in ['instant', 'cnt', 'date', 'hum', 'windspeed', 'dataset loaded']:
        continue  # Skip specified columns

    if data[col].dtype in ['object', 'category'] or col == 'season':
        # Use the actual names for categorical or object columns
        unique_values = data[col].dropna().unique()
        selected_values[col] = st.selectbox(f"Select value for {col}", options=unique_values, key=col)
    elif data[col].dtype in ['int64', 'float64']:
        # Provide a range of numbers as options
        unique_values = sorted(data[col].dropna().unique())
        selected_values[col] = st.selectbox(f"Select value for {col}", options=unique_values, key=col)

# Date selection
st.write("### Select Date")
if 'dteday' in data.columns:
    selected_date = st.date_input("Select a date", min_value=data['dteday'].min(), max_value=data['dteday'].max())

st.markdown("</div>", unsafe_allow_html=True)

# Predict button
if st.button("Predict"):
    # Filter data by selected values and date
    filtered_data = data[data['dteday'] == pd.Timestamp(selected_date)]
    for col, value in selected_values.items():
        filtered_data = filtered_data[filtered_data[col] == value]

    if not filtered_data.empty:
        total_rentals = filtered_data['cnt'].sum()
        st.write(f"## Total Bikes Rented on {selected_date}: {total_rentals}")
    else:
        st.write(f"## No bikes rented on {selected_date} for the selected conditions.")