import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("co2_emissions (1).csv")

# Check required columns
required_columns = ["make", "model", "engine_size", "cylinders", "fuel_consumption_comb(l/100km)", "fuel_consumption_city", "fuel_consumption_hwy", "co2_emissions"]
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise ValueError(f"Dataset is missing the following columns: {missing_columns}")

# Encode categorical variables
le_make = LabelEncoder()
le_model = LabelEncoder()

data["make"] = le_make.fit_transform(data["make"])
data["model"] = le_model.fit_transform(data["model"])

# Save the encoders
joblib.dump(le_make, "make_encoder.pkl")
joblib.dump(le_model, "model_encoder.pkl")

# Select features and target
features = ["make", "model", "engine_size", "cylinders", "fuel_consumption_comb(l/100km)", "fuel_consumption_city", "fuel_consumption_hwy"]
target = "co2_emissions"

X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "co2_model.pkl")

print("✅ Model training complete and saved as co2_model.pkl")
