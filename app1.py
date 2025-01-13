import streamlit as st
import pandas as pd
import io
import subprocess
import sys

# Function to ensure `nbformat` is installed
def ensure_nbformat_installed():
    try:
        import nbformat as nbf
        return True
    except ImportError:
        st.warning("`nbformat` not found. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat"])
        try:
            import nbformat as nbf
            return True
        except ImportError:
            st.error("Failed to install `nbformat`. Please install it manually using `pip install nbformat`.")
            return False

# Check if nbformat is installed
nbformat_installed = ensure_nbformat_installed()

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
    if col in ['instant', 'cnt']:
        continue  # Skip 'instant' and 'cnt' columns
    
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

    # Option to download filtered data as CSV
    csv = filtered_data.to_csv(index=False)
    st.download_button("Download Data as CSV", csv, "filtered_data.csv", "text/csv")

    # Option to download filtered data as Jupyter Notebook
    if nbformat_installed:
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_code_cell(f"""import pandas as pd

filtered_data = pd.DataFrame({filtered_data.to_dict()})
print(filtered_data)"""))

        # Convert the notebook to a file-like object (as bytes)
        nb_file = io.BytesIO()
        nbformat_data = nbf.writes(nb)  # Serialize the notebook into a string format
        nb_file.write(nbformat_data.encode('utf-8'))  # Encode as bytes before writing to the BytesIO object
        nb_file.seek(0)

        # Allow download of the notebook
        st.download_button("Download as Jupyter Notebook", nb_file, "filtered_data.ipynb", "application/x-ipynb+json")
    else:
        st.error("`nbformat` is still not available. Notebook download is disabled.")
