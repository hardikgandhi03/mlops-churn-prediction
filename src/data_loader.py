import pandas as pd
from sklearn.model_selection import train_test_split

def load_clean_data(filepath="data/cleaned.csv"):
    return pd.read_csv(filepath)

def preprocess(df):
    # One-hot encoding (drop_first=True to avoid multicollinearity)
    X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return train_test_split(X, y, test_size=0.2, random_state=42)
