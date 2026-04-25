import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return

    mlruns_dir = BASE_DIR / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())


df = pd.read_csv(BASE_DIR / "DataOps" / "Statics" / "irrigation_prediction_processed.csv")
y = df["Irrigation_Need"]
X = df.drop("Irrigation_Need", axis=1)

print("Classes presentes:", y.unique())

model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
features = joblib.load(BASE_DIR / "models" / "features.pkl")

X = X.reindex(columns=features, fill_value=0)
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

acc = accuracy_score(y, preds)
f1 = f1_score(y, preds, average="weighted")

print("\nAccuracy:", acc)
print("F1-score:", f1)
print("\nClassification Report:")
print(classification_report(y, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y, preds))

print("\nTest sample:")
print("Real:", y.iloc[0])
print("Pred:", preds[0])

configure_mlflow()
mlflow.set_experiment("Irrigation_MLOps_Experiment")
with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

print("\nEvaluation terminee")
