import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import numpy as np
import yaml

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "DataOps" / "Statics" / "irrigation_prediction_processed.csv"
PARAMS_PATH = BASE_DIR / "params.yml"
MODELS_DIR = BASE_DIR / "models"


def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")

    mlflow.set_experiment("Irrigation_MLOps_Experiment")


with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


df = pd.read_csv(DATA_PATH)

df = df[df["Irrigation_Need"] != 2].copy()

if "Irrigation_Need" not in df.columns:
    raise ValueError("Target column 'Irrigation_Need' not found")

y = df["Irrigation_Need"].astype(int)
X = df.drop("Irrigation_Need", axis=1)

print("\n Class distribution:")
print(y.value_counts())
print(" Classes mapping: [0=Low, 1=Medium]")

test_size = params["training"]["test_size"]
random_state = params["training"]["random_state"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y,
)

val_size = 0.2

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=val_size,
    random_state=random_state,
    stratify=y_train_val,
)

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(**params["models"]["LogisticRegression"]))
    ]),
    "RandomForest": Pipeline([
        ("model", RandomForestClassifier(**params["models"]["RandomForest"]))
    ]),
    "GradientBoosting": Pipeline([
        ("model", GradientBoostingClassifier(**params["models"]["GradientBoosting"]))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, **params["models"]["SVM"]))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(**params["models"]["KNN"]))
    ]),
}

best_model = None
best_score = 0.0
best_name = ""

configure_mlflow()

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        print(f"\nMODEL {name}")
        print("Accuracy:", acc)
        print("F1-score:", f1)
        print(classification_report(y_test, preds))

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, name="model")

        if f1 > best_score:
            best_model = model
            best_score = f1
            best_name = name


print("\n Sample prediction test:")
sample = X_test.iloc[[0]]
print("Prediction:", best_model.predict(sample)[0])
print("Real:", y_test.iloc[0])


MODELS_DIR.mkdir(parents=True, exist_ok=True)

scaler = best_model.named_steps.get("scaler", None)
joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

if scaler is not None:
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(" Scaler saved from pipeline")
else:
    # créer un scaler "dummy" entraîné
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(np.zeros((1, X.shape[1])))
    joblib.dump(dummy_scaler, MODELS_DIR / "scaler.pkl")
    print(" Dummy scaler created and saved")

joblib.dump(X.columns.tolist(), MODELS_DIR / "features.pkl")
joblib.dump([0, 1], MODELS_DIR / "classes.pkl")

with mlflow.start_run(run_name="BEST_MODEL"):
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_f1", best_score)

    try:
        mlflow.sklearn.log_model(
            best_model,
            name="best_model",
            registered_model_name="Irrigation_Best_Model",
        )
    except Exception as e:
        print("Registry unavailable:", e)
        mlflow.sklearn.log_model(best_model, name="best_model")

print("\n Training finished")
print(" Best model:", best_name)
print(" Best F1:", best_score)
print(" Classes: [0=Low, 1=Medium]")