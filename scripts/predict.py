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

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = joblib.load(FEATURES_PATH)
CLASSES = joblib.load(CLASSES_PATH)

# =========================
# 🧾 INPUT MODEL
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
    return {"status": "API running 🚀"}

# =========================
# 🔥 PREDICT
# =========================
@app.post("/predict")
def predict(data: InputData):

    try:
        # 1. dict input
        input_dict = data.model_dump()

        # 2. dataframe
        df = pd.DataFrame([input_dict])

        # 3. FORCE correct feature order (CRITICAL FIX)
        df = df.reindex(columns=FEATURES, fill_value=0)

        # 4. scaling
        X_scaled = scaler.transform(df)

        # 5. prediction
        pred = model.predict(X_scaled)[0]

        # 6. optional probability (if supported)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled).max()

        return {
            "prediction": str(pred),
            "confidence": float(prob) if prob is not None else None,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e)
        }