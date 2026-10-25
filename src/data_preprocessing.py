
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"

NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Age"]
CATEGORICAL_FEATURES = ["Location", "Property_Type"]
TARGET = "Price"

REQUIRED_COLUMNS = [
    "Property_ID",
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    TARGET,
]


def load_and_validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Property_ID"].isna().any():
        raise ValueError("Property_ID cannot contain missing values.")

    if df["Property_ID"].duplicated().any():
        raise ValueError("Property_ID values must be unique.")

    if df["Price"].isna().any() or (df["Price"] <= 0).any():
        raise ValueError("Price must be present and greater than zero.")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")

        if df[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values.")

        if not df[column].map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        ).all():
            raise ValueError(f"{column} contains invalid numeric values.")

    if (df["Area"] <= 0).any():
        raise ValueError("Area must be greater than zero.")

    if (df[["Bedrooms", "Bathrooms", "Age"]] < 0).any().any():
        raise ValueError("Bedrooms, bathrooms, and age cannot be negative.")

    for column in CATEGORICAL_FEATURES:
        if df[column].isna().any():
            raise ValueError(f"{column} cannot contain missing values.")

        df[column] = df[column].astype(str).str.strip()

        if df[column].eq("").any():
            raise ValueError(f"{column} cannot be empty.")

    return df


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def prepare_features(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


if __name__ == "__main__":
    data = load_and_validate_data()
    X, y = prepare_features(data)

    print("Dataset validated successfully.")
    print(f"Rows: {len(data)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Target: {TARGET}")
    print("\nPreprocessor configuration:")
    print(build_preprocessor())
