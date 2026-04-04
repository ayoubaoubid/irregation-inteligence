import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 📂 Load params
with open("irregation-inteligence/params.yml", "r") as f:
    params = yaml.safe_load(f)

# 📂 Load data
df = pd.read_csv("irregation-inteligence/DataOps/Statics/irrigation_prediction_processed.csv")

mapping = {0: "High", 1: "Low", 2: "Medium"}
y = df["y"].map(mapping)

X = df.drop("y", axis=1)

# ✂️ Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["training"]["test_size"],
    random_state=params["training"]["random_state"]
)

# ⚙️ Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Models + params
models = {
    "LogisticRegression": LogisticRegression(**params["models"]["LogisticRegression"]),
    "RandomForest": RandomForestClassifier(**params["models"]["RandomForest"]),
    "GradientBoosting": GradientBoostingClassifier(**params["models"]["GradientBoosting"]),
    "SVM": SVC(**params["models"]["SVM"]),
    "KNN": KNeighborsClassifier(**params["models"]["KNN"])
}

best_model = None
best_score = 0
best_name = ""

# 🧠 MLflow experiment
mlflow.set_experiment("Irrigation_MLOps_Experiment")

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_name", name)
        mlflow.log_params(params["models"][name])
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print(f"{name} -> {acc:.4f}")

        if acc > best_score:
            best_model = model
            best_score = acc
            best_name = name

# =========================
# 🏆 SAVE BEST MODEL
# =========================
joblib.dump(best_model, "irregation-inteligence/models/best_model.pkl")
joblib.dump(scaler, "irregation-inteligence/models/scaler.pkl")
joblib.dump(X.columns.tolist(), "irregation-inteligence/models/features.pkl")
joblib.dump(best_model.classes_, "irregation-inteligence/models/classes.pkl")

# =========================
# 🧠 FINAL LOG
# =========================
with mlflow.start_run(run_name="BEST_MODEL"):
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_accuracy", best_score)
    mlflow.sklearn.log_model(best_model, "best_model")

print("Training finished ")
print("Best model:", best_name)
print("Best accuracy:", best_score)
print("Classes:", best_model.classes_)