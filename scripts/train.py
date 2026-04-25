import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parents[1]


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return

    mlruns_dir = BASE_DIR / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())


with open(BASE_DIR / "params.yml", "r", encoding="utf-8") as handle:
    params = yaml.safe_load(handle)


df = pd.read_csv(BASE_DIR / "DataOps" / "Statics" / "irrigation_prediction_processed.csv")

le = LabelEncoder()
y = le.fit_transform(df["Irrigation_Need"])
X = df.drop("Irrigation_Need", axis=1)

print("Class distribution:")
print(pd.Series(y).value_counts())
print("Classes mapping:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"],
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(**params["models"]["LogisticRegression"]),
    "RandomForest": RandomForestClassifier(**params["models"]["RandomForest"]),
    "GradientBoosting": GradientBoostingClassifier(**params["models"]["GradientBoosting"]),
    "SVM": SVC(probability=True, **params["models"]["SVM"]),
    "KNN": KNeighborsClassifier(**params["models"]["KNN"]),
}

best_model = None
best_score = 0.0
best_name = ""

configure_mlflow()
mlflow.set_experiment("Irrigation_MLOps_Experiment")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        print(name)
        print("Accuracy:", acc)
        print("F1-score:", f1)
        print(classification_report(y_test, preds))

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        if f1 > best_score:
            best_model = model
            best_score = f1
            best_name = name

print("Sample prediction test:")
sample = X_test[0:1]
print("Prediction:", best_model.predict(sample)[0])
print("Real:", y_test[0])

models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, models_dir / "best_model.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")
joblib.dump(X.columns.tolist(), models_dir / "features.pkl")
joblib.dump(le.classes_, models_dir / "classes.pkl")

with mlflow.start_run(run_name="BEST_MODEL"):
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_f1", best_score)
    try:
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            registered_model_name="Irrigation_Best_Model",
        )
    except Exception as exc:
        print(f"Model registry unavailable, logging artifact only: {exc}")
        mlflow.sklearn.log_model(best_model, "best_model")

print("Training finished")
print("Best model:", best_name)
print("Best F1:", best_score)
print("Classes:", le.classes_)
