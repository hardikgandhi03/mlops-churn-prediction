import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_clean_data, preprocess

def train():
    df = load_clean_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", acc)

        report = classification_report(y_test, preds)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        mlflow.sklearn.log_model(model, "model")
        print(f"✅ Accuracy: {acc:.4f}")
        print("✅ Model and report logged to MLflow")

if __name__ == "__main__":
    train()
