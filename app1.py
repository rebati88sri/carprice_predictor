import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration for a professional look
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Custom CSS for "Moderate Attractive" Design
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    h1 {
        color: #1E1E1E;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .prediction-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load the Model ---
# Note: Ensure the path is correct or the file is in the same folder as this script
try:
    model_path = r'C:\ML_campusX\carprice_predictor\car_price_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please check the path.")
    st.stop()

# --- Header ---
st.title("🚗 Car Price Prediction Tool")
st.write("Fill in the details below to estimate the market value of your vehicle.")

# --- Form UI ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        car_model = st.selectbox("Car Model", 
            ["Hyundai", "Tata", "Mahindra", "BMW", "Honda", "Audi", "Toyota", "Kia", "Maruti", "Ford"])
        
        engine_condition = st.selectbox("Engine Condition", 
            ["Excellent", "Good", "Average", "Poor"])
        
        fuel_type = st.selectbox("Fuel Type", 
            ["Petrol", "Diesel", "Hybrid", "Electric"])
        
        reparation = st.selectbox("Reparation Work Done?", ["No", "Yes"])

    with col2:
        purchase_val = st.number_input("Original Purchase Value ($)", min_value=0, value=10000, step=500)
        km_driven = st.number_input("Kilometers Driven", min_value=0, value=5000, step=1000)
        car_age = st.number_input("Car Age (Years)", min_value=0, max_value=50, value=2)
        no_buyers = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=0)

# --- Prediction Logic ---
if st.button("Predict Car Price"):
    # Pre-processing: Convert inputs to match model format
    # Note: You may need to adjust these mappings to match how you trained your model
    
    # Example Mapping (Encoding categorical data)
    model_mapping = {"BMW": 0, "Audi": 1, "Toyota": 2} # Replace with your actual encoding
    
    # Create input DataFrame (order must match model's expected features)
    input_data = pd.DataFrame([[
        car_model, engine_condition, fuel_type, reparation, 
        purchase_val, no_buyers, km_driven, car_age
    ]], columns=['car_model', 'engine_condition', 'fuel_type', 'reparation', 
                 'purchase_value', 'no_of_buyers', 'km_running', 'car_age'])

    # Prediction
    try:
        prediction = model.predict(input_data)
        
        # Display Result
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color: #4CAF50;'>Estimated Resale Price</h3>
                <h1 style='color: #1E1E1E;'>${round(prediction[0], 2):,}</h1>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()
    except Exception as e:
        st.error(f"Error in prediction: {e}. Check if feature names match your model's training data.")