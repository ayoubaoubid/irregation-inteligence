from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import csv
from datetime import datetime
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from fastapi import Request
import traceback

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "features.pkl")

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "predictions.csv")
NEW_DATA_PATH = os.path.join(LOG_DIR, "new_data.csv")

os.makedirs(LOG_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)

# CREATE new_data.csv if not exists
if not os.path.exists(NEW_DATA_PATH):
    with open(NEW_DATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURES) 

CLASS_MAPPING = {0: "Low", 1: "Medium"}

# PROMETHEUS
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
PREDICTION_COUNT = Counter("predictions_total", "Total Predictions", ["prediction_class"])
CONFIDENCE_GAUGE = Gauge("model_confidence", "Prediction Confidence")


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


@app.get("/")
def home():
    return {"status": "API running"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/alert")
async def alert(request: Request):
    data = await request.json()

    print(" ALERT RECEIVED FROM ALERTMANAGER:")
    print(data)

    # option: log file
    with open("logs/alerts.log", "a") as f:
        f.write(str(data) + "\n")

    return {"status": "alert received"}

@app.post("/predict")
def predict(data: InputData):

    REQUEST_COUNT.inc()

    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        # validate columns
        missing = set(FEATURES) - set(df.columns)
        extra = set(df.columns) - set(FEATURES)

        if missing:
            return {"error": f"Missing features: {missing}"}

        df = df[FEATURES]

        # 🔥 prediction
        pred = int(model.predict(df)[0])

        # 🔥 store raw data for drift detection
        with open(NEW_DATA_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(df.values.flatten())

        confidence = 1.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            confidence = float(max(probs))
            CONFIDENCE_GAUGE.set(confidence)

        label = CLASS_MAPPING.get(pred, str(pred))
        PREDICTION_COUNT.labels(prediction_class=label).inc()

        # log predictions
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), pred, label, confidence])

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}