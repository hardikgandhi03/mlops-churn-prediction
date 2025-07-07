import pandas as pd
import mlflow.sklearn
from typing import Union
from pydantic import BaseModel

def load_model():
    # Loads the saved model from the local "model" directory
    return mlflow.sklearn.load_model("model")

def preprocess_input(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Preprocess the input DataFrame to match the trained model's expected features.
    Applies one-hot encoding and aligns missing columns.
    """
    df_encoded = pd.get_dummies(df)

    # Ensure input columns match model's training columns
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Fill missing with 0

    return df_encoded[model_columns]

def make_prediction(model, input_data: Union[dict, BaseModel]) -> bool:
    """
    Make a prediction given the model and raw input data (dict or Pydantic model).
    """
    if isinstance(input_data, BaseModel):
        input_data = input_data.model_dump()  # ✅ Use model_dump() for Pydantic v2

    df = pd.DataFrame([input_data])
    processed = preprocess_input(df, model)
    prediction = model.predict(processed)[0]
    return bool(prediction)