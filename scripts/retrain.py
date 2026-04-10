import os
import time

DATA_PATH = "irregation-inteligence/DataOps/Statics/irrigation_prediction_processed.csv"

last_version = os.path.getmtime(DATA_PATH)

while True:
    current_version = os.path.getmtime(DATA_PATH)

    if current_version != last_version:
        print("New data detected → retraining...")
        os.system("python scripts/train.py")
        last_version = current_version
    else :
        print("No new data. Checking again in 60 seconds...")
    time.sleep(60)