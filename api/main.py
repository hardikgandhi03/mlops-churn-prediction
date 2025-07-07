from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from api.utils import load_model, preprocess_input

app = FastAPI()
model = load_model()

# Define your input schema using Pydantic for validation
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: CustomerData):
    try:
        print("🚀 Incoming data:", data.model_dump())  # ← Debug log
        df = pd.DataFrame([data.model_dump()])
        processed = preprocess_input(df, model)
        prediction = model.predict(processed)
        return {"churn": bool(prediction[0])}
    except Exception as e:
        print("❌ Error during prediction:", str(e))  # ← Print the real error
        raise e
