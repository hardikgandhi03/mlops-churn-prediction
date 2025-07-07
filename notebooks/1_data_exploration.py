import pandas as pd
import os

# Load dataset
RAW_PATH = "raw_data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = "data/cleaned.csv"

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Read CSV
df = pd.read_csv(RAW_PATH)

# Replace empty strings in 'TotalCharges' and convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", pd.NA), errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Drop customerID column
df = df.drop(columns=["customerID"])

# Convert target to binary
df['Churn'] = df['Churn'].map({"Yes": 1, "No": 0})

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Cleaned data written to: {OUTPUT_PATH}")
