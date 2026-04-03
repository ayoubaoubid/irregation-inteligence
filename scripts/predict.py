from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI()

# =========================
# 📁 PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# 📊 FEATURES ORDER (IMPORTANT)
# =========================
FEATURES = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
    "Crop_Growth_Stage_Harvest",
    "Crop_Growth_Stage_Sowing",
    "Crop_Growth_Stage_Vegetative",
    "Irrigation_Type_Drip",
    "Irrigation_Type_Rainfed",
    "Irrigation_Type_Sprinkler",
    "Mulching_Used_Yes"
]

# =========================
# 🧾 INPUT VALIDATION (Pydantic)
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
    Crop_Growth_Stage_Harvest: float
    Crop_Growth_Stage_Sowing: float
    Crop_Growth_Stage_Vegetative: float
    Irrigation_Type_Drip: float
    Irrigation_Type_Rainfed: float
    Irrigation_Type_Sprinkler: float
    Mulching_Used_Yes: float


# =========================
# 🟢 HOME
# =========================
@app.get("/")
def home():
    return {"status": "API running 🚀"}

# =========================
# 🔥 PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(data: InputData):

    try:
        # 1️⃣ convert input to dict
        input_data = data.model_dump()

        # 2️⃣ dataframe
        df = pd.DataFrame([input_data])

        # 3️⃣ ensure correct column order
        df = df.reindex(columns=FEATURES)

        # 4️⃣ scaling
        scaled_data = scaler.transform(df)

        # 5️⃣ prediction
        prediction = model.predict(scaled_data)

        return {
            "prediction": str(prediction[0]),
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e)
        }