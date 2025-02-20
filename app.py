import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
st.title("CO2 Emission Prediction App 🚗💨")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset
    st.write("### Preview of Dataset")
    st.write(df.head())

    # Check for column names
    st.write("### Column Names")
    st.write(df.columns.tolist())

    # Handling categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    # Drop rows with missing values
    df = df.dropna()

    if "co2_emissions" not in df.columns:
        st.error("❌ Column 'co2_emissions' not found in dataset! Please check your file.")
    else:
        X = df.drop(columns=["co2_emissions"])  # Features
        y = df["co2_emissions"]  # Target

        # Encode categorical features
        X_encoded = pd.get_dummies(X, columns=categorical_cols)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model evaluation
        st.write("### Model Performance")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # Feature importance
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        st.write("### Feature Importance")
        st.bar_chart(feature_importances.sort_values(ascending=False))

        # User Input for Prediction
        st.write("## Predict CO2 Emission")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.text_input(f"Enter {col}", "")

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])

            # Encode input data
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

            # Prediction
            prediction = model.predict(input_encoded)[0]
            st.success(f"Predicted CO2 Emission: {prediction:.2f} g/km")
