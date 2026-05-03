from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "DataOps" / "Statics"

IMPORTANT_COLUMNS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Crop_Growth_Stage",
    "Irrigation_Type",
    "Field_Area_hectare",
    "Mulching_Used",
    "Previous_Irrigation_mm",
    "Irrigation_Need",
]

NUMERIC_COLUMNS = [
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
]

CATEGORICAL_COLUMNS = [
    "Crop_Growth_Stage",
    "Irrigation_Type",
    "Mulching_Used",
]

TARGET_COLUMN = "Irrigation_Need"
TARGET_MAPPING = {"Low": 0.0, "Medium": 1.0, "High": 2.0}


def oversample_majority_size(
    features: pd.DataFrame, target: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    combined = features.copy()
    combined[TARGET_COLUMN] = target

    grouped = []
    target_counts = combined[TARGET_COLUMN].value_counts()
    majority_size = int(target_counts.max())

    for class_value in sorted(target_counts.index):
        class_rows = combined[combined[TARGET_COLUMN] == class_value]
        if len(class_rows) < majority_size:
            sampled = class_rows.sample(
                n=majority_size, replace=True, random_state=random_state
            )
            grouped.append(sampled)
        else:
            grouped.append(class_rows)

    balanced = (
        pd.concat(grouped, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    return balanced.drop(columns=[TARGET_COLUMN]), balanced[TARGET_COLUMN]


def preprocess_dataframe(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    important = df.loc[:, IMPORTANT_COLUMNS].copy()
    important = important.dropna().reset_index(drop=True)

    numeric_part = important[NUMERIC_COLUMNS].copy()
    categorical_part = important[CATEGORICAL_COLUMNS].copy()
    target = important[TARGET_COLUMN].map(TARGET_MAPPING)

    unknown_labels = important.loc[target.isna(), TARGET_COLUMN].unique().tolist()
    if unknown_labels:
        raise ValueError(f"Unexpected target labels: {unknown_labels}")

    scaled_numeric = pd.DataFrame(
        StandardScaler().fit_transform(numeric_part),
        columns=NUMERIC_COLUMNS,
        index=important.index,
    )
    encoded_categorical = pd.get_dummies(categorical_part)

    processed_features = pd.concat([scaled_numeric, encoded_categorical], axis=1)
    processed_features, balanced_target = oversample_majority_size(
        processed_features, target, random_state=random_state
    )

    processed = processed_features.copy()
    processed[TARGET_COLUMN] = balanced_target.astype(float)

    return important, processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the ML-ready processed irrigation dataset."
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "irrigation_prediction_Variables_Important.csv"),
        help="Path to the selected important-feature irrigation CSV dataset.",
    )
    parser.add_argument(
        "--processed-output",
        default=str(DATA_DIR / "irrigation_prediction_processed.csv"),
        help="Path to save the ML-ready processed dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for oversampling and row shuffling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    processed_output = Path(args.processed_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    processed_output.parent.mkdir(parents=True, exist_ok=True)

    important_df = pd.read_csv(input_path)
    _, processed_df = preprocess_dataframe(
        important_df, random_state=args.random_state
    )

    processed_df.to_csv(processed_output, index=False)

    print(f"Loaded important dataset: {input_path} ({important_df.shape[0]} rows)")
    print(f"Saved processed dataset: {processed_output} ({processed_df.shape[0]} rows)")
    print(f"Processed columns: {processed_df.columns.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
