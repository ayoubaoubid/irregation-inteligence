import pandas as pd
import os
import yaml
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Load parameters
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yml")
with open(params_path, "r") as f:
    params = yaml.safe_load(f)


# Load dataset
df = pd.read_csv("DataOps/Statics/irrigation_prediction_processed.csv")

# Encode target
le = LabelEncoder()
y = le.fit_transform(df["Irrigation_Need"])

X = df.drop("Irrigation_Need", axis=1)

print("Class distribution:")
print(pd.Series(y).value_counts())

print("Classes mapping:", le.classes_)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"],
    stratify=y
)


# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Models
models = {
    "LogisticRegression": LogisticRegression(**params["models"]["LogisticRegression"]),
    "RandomForest": RandomForestClassifier(**params["models"]["RandomForest"]),
    "GradientBoosting": GradientBoostingClassifier(**params["models"]["GradientBoosting"]),
    "SVM": SVC(probability=True, **params["models"]["SVM"]),
    "KNN": KNeighborsClassifier(**params["models"]["KNN"])
}


best_model = None
best_score = 0
best_name = ""

mlflow.set_experiment("Irrigation_MLOps_Experiment")


# Training loop
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


# Quick test
print("Sample prediction test:")
sample = X_test[0:1]
print("Prediction:", best_model.predict(sample)[0])
print("Real:", y_test[0])


# Save artifacts
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")
joblib.dump(le.classes_, "models/classes.pkl")


# Final MLflow log
with mlflow.start_run(run_name="BEST_MODEL"):
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_f1", best_score)
    mlflow.sklearn.log_model(best_model, "best_model",registered_model_name="Irrigation_Best_Model")

print("Training finished")
print("Best model:", best_name)
print("Best F1:", best_score)
print("Classes:", le.classes_)