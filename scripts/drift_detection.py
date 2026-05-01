import pandas as pd
import numpy as np

def psi(expected, actual, buckets=10):

    def scale_range(arr):
        return np.histogram(arr, bins=buckets)[0] / len(arr)

    expected_perc = scale_range(expected)
    actual_perc = scale_range(actual)

    psi_value = np.sum(
        (expected_perc - actual_perc) *
        np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )

    return psi_value


def monitor():
    baseline = pd.read_csv("DataOps/Statics/irrigation_prediction_processed.csv")
    new_data = pd.read_csv("logs/new_data.csv")

    print("🔍 Drift Monitoring Running...\n")

    for col in baseline.columns:
        if col in new_data.columns:
            score = psi(baseline[col], new_data[col])
            print(f"{col}: PSI = {score:.4f}")

            if score > 0.2:
                print("🚨 DRIFT DETECTED on:", col)