# Importing libraries-----------------------------------------------------------------------------------------
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# Creating Sidebar-------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# CO2 Emissions by Vehicle")
    user_input = st.selectbox('Please select',('Visualization','Model'))

# Load the vehicle dataset
try:
    df = pd.read_csv('co2 Emissions.csv')
    df.columns = df.columns.str.strip()  # Remove extra spaces
except FileNotFoundError:
    st.error("Dataset not found. Please check the file path.")
    st.stop()

# Drop rows with natural gas as fuel type
fuel_type_mapping = {"Z": "Premium Gasoline","X": "Regular Gasoline","D": "Diesel","E": "Ethanol(E85)","N": "Natural Gas"}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping).fillna('Unknown')
df_natural = df[df["Fuel Type"] != "Natural Gas"].reset_index(drop=True)

# Remove outliers from the data
df_new = df_natural[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
df_new = df_new.dropna()
df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]

# Visualization-------------------------------------------------------------------------------------------------
if user_input == 'Visualization':

    # Remove unwanted warnings---------------------------------------------------------------------------------
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Showing Dataset------------------------------------------------------------------------------------------
    st.title('CO2 Emissions by Vehicle')
    st.header("Sample of Collected Data")
    st.dataframe(df.head())

    # Brands of Cars-------------------------------------------------------------------------------------------
    st.subheader('Brands of Cars')
    df_brand = df['Make'].value_counts().reset_index()
    df_brand.columns = ['Make', 'Count']
    fig1, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(data=df_brand, x="Make", y="Count", ax=ax)
    plt.xticks(rotation=75)
    plt.title("All Car Companies and their Cars")
    plt.xlabel("Companies")
    plt.ylabel("Cars")
    ax.bar_label(ax.containers[0], fontsize=7)
    st.pyplot(fig1)
    st.dataframe(df_brand)

    # Fix other similar sections by ensuring correct `rename(columns=...)` syntax, passing `fig` to `st.pyplot(fig)`
    # ... (rest of the visualization sections follow the same structure)

# Model-------------------------------------------------------------------------------------------------
else:
    # Prepare the data for modeling--------------------------------------------------------------------
    X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_new_model['CO2 Emissions(g/km)']

    # Train the random forest regression model---------------------------------------------------------
    model = RandomForestRegressor(random_state=42).fit(X, y)

    # Create the Streamlit web app---------------------------------------------------------------------
    st.title('CO2 Emission Prediction')
    st.write('Enter the vehicle specifications to predict CO2 emissions.')

    # Input fields for user----------------------------------------------------------------------------
    engine_size = st.number_input('Engine Size(L)', step=0.1, format="%.1f", min_value=0.1)
    cylinders = st.number_input('Cylinders', min_value=2, max_value=16, step=1)
    fuel_consumption = st.number_input('Fuel Consumption Comb (L/100 km)', step=0.1, format="%.1f", min_value=0.1)

    if st.button('Predict'):
        try:
            input_data = np.array([[engine_size, cylinders, fuel_consumption]])
            predicted_co2 = model.predict(input_data)
            st.write(f'Predicted CO2 Emissions: {predicted_co2[0]:.2f} g/km')
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
