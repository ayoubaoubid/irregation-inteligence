import os
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "DataOps" / "Statics" / "irrigation_prediction_processed.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

current_version = os.path.getmtime(DATA_PATH)

STATE_FILE = BASE_DIR / "models" / ".data_version.txt"

if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as f:
        last_version = float(f.read().strip())
else:
    last_version = None

if last_version is None or current_version != last_version:
    print("New data detected -> retraining...")

    subprocess.run(["python", "scripts/train.py"], check=True)

    os.makedirs("models", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        f.write(str(current_version))

else:
    print("No new data -> nothing to do.")