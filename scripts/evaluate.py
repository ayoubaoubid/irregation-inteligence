import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score

df = pd.read_csv("irregation-inteligence/DataOps/Statics/irrigation_prediction_processed.csv")

X = df.drop("Irrigation_Need", axis=1)
y = df["Irrigation_Need"]

model = joblib.load("irregation-inteligence/models/best_model.pkl")

preds = model.predict(X)

acc = accuracy_score(y, preds)

print("Accuracy:", acc)

# MLflow log
mlflow.set_experiment("Irrigation_MLOps_Experiment")

with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metric("final_accuracy", acc)