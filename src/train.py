import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os

# Read data
df = pd.read_csv("data/cleaned.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Set experiment
mlflow.set_tracking_uri("file:///mlruns")
mlflow.set_experiment("churn-prediction")

# Log model locally
with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")  # Logs to ./mlruns by default
    mlflow.sklearn.save_model(model, path="model")  # ✅ Explicitly save to ./model

print("✅ Model saved to ./model/")

report = classification_report(y_test, y_pred)

# Save to text file
with open("classification_report.txt", "w") as f:
    f.write(report)