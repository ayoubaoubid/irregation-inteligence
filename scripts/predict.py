from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# =========================
# 📁 PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.pkl")

# =========================
# 📦 LOAD MODELS
# =========================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = joblib.load(FEATURES_PATH)
CLASSES = joblib.load(CLASSES_PATH)

print(" Model loaded")
print(" FEATURES :", FEATURES)
print(" CLASSES :", CLASSES)

# =========================
# 🧾 INPUT SCHEMA
# =========================
class InputData(BaseModel):
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Temperature_C: float
    Humidity: float
    Rainfall_mm: float
    Sunlight_Hours: float
    Wind_Speed_kmh: float
    Field_Area_hectare: float
    Previous_Irrigation_mm: float

    Crop_Growth_Stage_Flowering: float
    Crop_Growth_Stage_Harvest: float
    Crop_Growth_Stage_Sowing: float
    Crop_Growth_Stage_Vegetative: float

    Irrigation_Type_Canal: float = 0
    Irrigation_Type_Drip: float = 0
    Irrigation_Type_Rainfed: float = 0
    Irrigation_Type_Sprinkler: float = 0

    Mulching_Used_No: float = 0
    Mulching_Used_Yes: float = 0


# =========================
# 🟢 HOME
# =========================
@app.get("/")
def home():
    return {"status": "API running "}


# =========================
# 🔥 PREDICT
# =========================
@app.post("/predict")
def predict(data: InputData):

    try:
        # =========================
        # 1. INPUT
        # =========================
        input_dict = data.model_dump()

        # =========================
        # 2. DATAFRAME
        # =========================
        df = pd.DataFrame([input_dict])

        print("\n📥 INPUT REÇU :")
        print(df)

        # =========================
        # 3. FEATURE CHECK
        # =========================
        missing_cols = set(FEATURES) - set(df.columns)
        extra_cols = set(df.columns) - set(FEATURES)

        if missing_cols:
            return {
                "error": f"Missing features: {missing_cols}",
                "status": "failed"
            }

        if extra_cols:
            print(f" Colonnes ignorées: {extra_cols}")

        # =========================
        # 4. ALIGN FEATURES
        # =========================
        df = df.reindex(columns=FEATURES, fill_value=0)

        print("\n FEATURES ALIGNED :")
        print(df)

        # =========================
        # 5. SCALE
        # =========================
        X_scaled = scaler.transform(df)

        # =========================
        # 6. PREDICT CLASS ID
        # =========================
        pred_class_id = int(model.predict(X_scaled)[0])

        # =========================
        # 7. PROBABILITIES
        # =========================
        confidence = None
        probs_dict = {}

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            confidence = float(max(probs))

            probs_dict = {
                str(cls): float(p)
                for cls, p in zip(model.classes_, probs)
            }

            print("\n Probabilités :")
            for k, v in probs_dict.items():
                print(f"{k} : {v:.4f}")

        # =========================
        # 8. MAP LABEL (IMPORTANT FIX)
        # =========================
        class_label = str(pred_class_id)

        if len(CLASSES) > 0:
            try:
                class_label = str(CLASSES[pred_class_id])
            except:
                class_label = str(pred_class_id)

        # =========================
        # 9. RESPONSE
        # =========================
        return {
            "prediction_id": pred_class_id,
            "prediction_label": class_label,
            "confidence": confidence,
            "probabilities": probs_dict,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }