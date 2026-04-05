import pandas as pd
import joblib
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("irregation-inteligence/DataOps/Statics/irrigation_prediction_processed.csv")

mapping = {0: "Low", 1: "Medium", 2: "High"}
y = df["Irrigation_Need"]
X = df.drop("Irrigation_Need", axis=1)

print(" Classes présentes :", y.unique())

# =========================
# 📦 LOAD MODEL + SCALER
# =========================
model = joblib.load("irregation-inteligence/models/best_model.pkl")
scaler = joblib.load("irregation-inteligence/models/scaler.pkl")
FEATURES = joblib.load("irregation-inteligence/models/features.pkl")

# =========================
# ⚠️ ALIGN FEATURES
# =========================
X = X.reindex(columns=FEATURES, fill_value=0)

# =========================
# ⚙️ SCALING (CRUCIAL)
# =========================
X_scaled = scaler.transform(X)

# =========================
# 🔥 PREDICTION
# =========================
preds = model.predict(X_scaled)

# =========================
# 📊 METRICS
# =========================
acc = accuracy_score(y, preds)
f1 = f1_score(y, preds, average="weighted")

print("\n Accuracy :", acc)
print(" F1-score :", f1)

print("\n Classification Report :")
print(classification_report(y, preds))

print("\n Confusion Matrix :")
print(confusion_matrix(y, preds))

# =========================
# 🧪 TEST RAPIDE (1 SAMPLE)
# =========================
print("\n TEST SAMPLE :")
print("Real :", y.iloc[15677])
print("Pred :", preds[15677])

# =========================
# 📊 MLflow LOG
# =========================
mlflow.set_experiment("Irrigation_MLOps_Experiment")

with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
print("\n Evaluation terminée")