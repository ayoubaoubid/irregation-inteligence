import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]


def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")

    mlflow.set_experiment("Irrigation_MLOps_Experiment")


df = pd.read_csv(BASE_DIR / "DataOps" / "Statics" / "irrigation_prediction_processed.csv")

df = df[df["Irrigation_Need"] != 2].copy()

y = df["Irrigation_Need"].astype(int)
X = df.drop("Irrigation_Need", axis=1)

print("Classes presentes:", y.unique())

model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
features = joblib.load(BASE_DIR / "models" / "features.pkl")

X = X.reindex(columns=features, fill_value=0)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# predict ONLY on test set
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="weighted")

print("\nAccuracy:", acc)
print("F1-score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\nTest sample:")
print("Real:", y_test.iloc[0])
print("Pred:", preds[0])

configure_mlflow()

with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

print("\nEvaluation terminee")