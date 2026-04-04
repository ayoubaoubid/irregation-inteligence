from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "DataOps" / "Statics"
DJANGO_DIR = ROOT_DIR / "Plateform_IA"

DATASETS = {
    "irrigation_prediction.csv": {
        "required_columns": {
         'Soil_pH',
         'Soil_Moisture',
         'Organic_Carbon',
         'Electrical_Conductivity',
         'Temperature_C',
         'Humidity',
         'Rainfall_mm',
         'Sunlight_Hours',
         'Wind_Speed_kmh',
         'Crop_Growth_Stage',
         'Irrigation_Type',
         'Field_Area_hectare',
         'Mulching_Used',
         'Previous_Irrigation_mm',
         'Irrigation_Need',
        }
    },
    "irrigation_prediction_processed.csv": {
        "required_columns": {
'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
       'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
       'Wind_Speed_kmh', 'Crop_Growth_Stage', 'Irrigation_Type',
       'Field_Area_hectare', 'Mulching_Used', 'Previous_Irrigation_mm',
       'Irrigation_Need',
        }
    },
    "irrigation_prediction_Variables_Important.csv": {
        "required_columns": {
'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
       'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
       'Wind_Speed_kmh', 'Crop_Growth_Stage', 'Irrigation_Type',
       'Field_Area_hectare', 'Mulching_Used', 'Previous_Irrigation_mm',
       'Irrigation_Need',
        }
    },
}


def validate_csv_file(csv_path: Path, required_columns: set[str]) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {csv_path}")

        available = set(reader.fieldnames)
        missing = required_columns - available
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path.name} is missing columns: {missing_list}")

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"{csv_path.name} is empty")


def run_command(command: list[str], cwd: Path) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate datasets and run Django quality checks."
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Run only dataset validation and Django configuration checks.",
    )
    args = parser.parse_args()

    print("Step 1/3 - validating DataOps datasets")
    for filename, config in DATASETS.items():
        validate_csv_file(DATA_DIR / filename, config["required_columns"])
        print(f"Validated: {filename}")

    print("Step 2/3 - running Django system checks")
    run_command([sys.executable, "manage.py", "check"], cwd=DJANGO_DIR)

    if args.skip_tests:
        print("Step 3/3 - tests skipped by request")
        return 0

    print("Step 3/3 - running Django test suite")
    run_command([sys.executable, "manage.py", "test"], cwd=DJANGO_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

