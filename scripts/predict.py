from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# =========================
# 📁 PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")

# =========================
# 📦 LOAD MODELS
# =========================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = joblib.load(FEATURES_PATH)

print("Model loaded successfully")
print("Features:", FEATURES)

# =========================
# 🔥 CLASS MAPPING (FIX IMPORTANT)
# =========================
CLASS_MAPPING = {
    0: "Low",
    1: "Medium",
    2: "High"
}

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
    return {"status": "API running"}

# =========================
# 🔥 PREDICT
# =========================
@app.post("/predict")
def predict(data: InputData):

    try:
        # 1. INPUT
        input_dict = data.model_dump()

        # 2. DATAFRAME
        df = pd.DataFrame([input_dict])

        print("\n INPUT RECEIVED:")
        print(df)

        # 3. CHECK FEATURES
        missing_cols = set(FEATURES) - set(df.columns)
        extra_cols = set(df.columns) - set(FEATURES)

        if missing_cols:
            return {
                "error": f"Missing features: {missing_cols}",
                "status": "failed"
            }

        if extra_cols:
            print(f"Extra columns ignored: {extra_cols}")

        # 4. ALIGN FEATURES
        df = df.reindex(columns=FEATURES, fill_value=0)

        print("\n ALIGNED FEATURES:")
        print(df)

        # 5. SCALE
        X_scaled = scaler.transform(df)

        # 6. PREDICT
        pred_class_id = int(model.predict(X_scaled)[0])

        # 7. PROBABILITIES
        probs_dict = {}
        confidence = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            confidence = float(max(probs))

            probs_dict = {
                str(i): float(p)
                for i, p in enumerate(probs)
            }

        # 8. CLASS LABEL (FIX HERE)
        class_label = CLASS_MAPPING.get(pred_class_id, str(pred_class_id))

        # 9. RESPONSE
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
