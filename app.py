import streamlit as st
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("co2_model.pkl")
le_make = joblib.load("make_encoder.pkl")
le_model = joblib.load("model_encoder.pkl")

# Streamlit UI
st.title("CO2 Emissions Prediction App🚗💨")


# User input fields
make = st.selectbox("Select Car Make", le_make.classes_)
model_name = st.selectbox("Select Car Model", le_model.classes_)
engine_size = st.number_input("Engine Size (L)", min_value=0.1, max_value=10.0, step=0.1)
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, step=1)
fuel_consumption_comb = st.number_input("Fuel Consumption (Combined) (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
fuel_consumption_city = st.number_input("Fuel Consumption (City) (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)
fuel_consumption_hwy = st.number_input("Fuel Consumption (Highway) (L/100 km)", min_value=1.0, max_value=30.0, step=0.1)

# Prediction function
def predict_co2_emissions(make, model_name, engine_size, cylinders, fuel_consumption_comb, fuel_consumption_city, fuel_consumption_hwy):
    # Encode categorical features
    make_encoded = le_make.transform([make])[0]
    model_encoded = le_model.transform([model_name])[0]

    # Create input array
    input_data = np.array([[make_encoded, model_encoded, engine_size, cylinders, fuel_consumption_comb, fuel_consumption_city, fuel_consumption_hwy]])

    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Predict button
if st.button("Predict CO2 Emissions"):
    prediction = predict_co2_emissions(make, model_name, engine_size, cylinders, fuel_consumption_comb, fuel_consumption_city, fuel_consumption_hwy)
    st.success(f"Predicted CO2 Emissions: {prediction:.2f} g/km")
