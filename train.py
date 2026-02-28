# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset from a working source
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("✅ Columns:", df.columns.tolist())  # Debug print

# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- AZURE ML SPECIFIC UPDATES BELOW ---

# Create the outputs folder if it doesn't exist
# (Azure ML automatically uploads everything in the ./outputs folder to your workspace)
os.makedirs('./outputs', exist_ok=True)

# Save the model inside the outputs folder
model_path = './outputs/diabetes_model.pkl'
joblib.dump(model, model_path)

print(f"✅ Model successfully saved to {model_path}")
