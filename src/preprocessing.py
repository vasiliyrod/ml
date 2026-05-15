"""
Data preprocessing module for Car Price Prediction.

This module handles:
- Loading raw data
- Cleaning (missing values, duplicates, outliers)
- Feature engineering
- Encoding categorical variables
- Train/val/test splitting
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib


def load_data(raw_path: str = "data/raw/cars_dataset_raw.csv") -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {raw_path}")

    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    # Rename columns to simpler names
    column_mapping = {
        "Company Names": "company",
        "Cars Names": "model",
        "Engines": "engine_type",
        "CC/Battery Capacity": "engine_capacity",
        "HorsePower": "horsepower",
        "Total Speed": "top_speed",
        "Performance(0 - 100 )KM/H": "acceleration_0_100",
        "Cars Prices": "price",
        "Fuel Types": "fuel_type",
        "Seats": "seats",
        "Torque": "torque",
    }

    df = df.rename(columns=column_mapping)
    return df


def extract_numeric_value(value) -> float:
    """Extract numeric value from string (e.g., '963 hp' -> 963, '$1,100,000' -> 1100000)."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()

    # Remove common units and symbols
    value_str = re.sub(r"[hp|km/h|sec|cc|Nm|\$|,]", "", value_str, flags=re.IGNORECASE)

    # Handle ranges (e.g., "70-85" -> take average)
    if "-" in value_str:
        try:
            parts = value_str.split("-")
            if len(parts) == 2:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2
        except ValueError:
            pass

    # Extract first number found
    match = re.search(r"[\d.]+", value_str)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan

    return np.nan


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns."""
    numeric_cols = [
        "engine_capacity",
        "horsepower",
        "top_speed",
        "acceleration_0_100",
        "price",
        "torque",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric_value)

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
        df[col] = df[col].fillna(mode_val)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    initial_shape = df.shape
    df = df.drop_duplicates()
    final_shape = df.shape

    print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features (feature engineering)."""
    # Power-to-weight ratio approximation (using seats as proxy for weight)
    if "horsepower" in df.columns and "seats" in df.columns:
        df["hp_per_seat"] = df["horsepower"] / df["seats"].replace(0, 1)

    # Engine capacity categories
    if "engine_capacity" in df.columns:
        df["engine_size_category"] = pd.cut(
            df["engine_capacity"],
            bins=[0, 1000, 2000, 3000, 4000, 6000, 10000],
            labels=["micro", "small", "medium", "large", "xl", "xxl"],
        )

    # Price category (for potential classification tasks)
    if "price" in df.columns:
        df["price_category"] = pd.cut(
            df["price"],
            bins=[0, 20000, 50000, 100000, 200000, 500000, float("inf")],
            labels=[
                "budget",
                "economy",
                "mid_range",
                "premium",
                "luxury",
                "ultra_luxury",
            ],
        )

    # Is electric or hybrid
    if "fuel_type" in df.columns:
        df["is_electric_hybrid"] = (
            df["fuel_type"]
            .str.lower()
            .isin(["electric", "plug in hyrbrid", "plug-in hybrid", "hybrid"])
            .astype(int)
        )

    # Performance score (combination of acceleration and top speed)
    if "acceleration_0_100" in df.columns and "top_speed" in df.columns:
        # Lower acceleration is better, higher top speed is better
        df["performance_score"] = df["top_speed"] / (
            df["acceleration_0_100"].replace(0, 1)
        )

    return df


def encode_categorical_variables(
    df: pd.DataFrame, fit: bool = True, encoder_path: str = None
) -> tuple:
    """Encode categorical variables using one-hot encoding."""
    categorical_cols = [
        "company",
        "fuel_type",
        "engine_type",
        "engine_size_category",
        "price_category",
        "model",
    ]

    # Filter to only existing columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    encoded_df = df.copy()
    encoders = {}

    if fit:
        # One-hot encoding
        encoded_df = pd.get_dummies(
            encoded_df, columns=categorical_cols, drop_first=True
        )

        if encoder_path:
            joblib.dump({"columns": categorical_cols}, encoder_path)
    else:
        # Load encoder info if needed for transformation
        if encoder_path and Path(encoder_path).exists():
            joblib.load(encoder_path)
            # Apply same encoding logic
            encoded_df = pd.get_dummies(
                encoded_df, columns=categorical_cols, drop_first=True
            )

    return encoded_df, encoders


def split_data(
    df: pd.DataFrame,
    target_col: str = "price",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Split data into train, validation, and test sets."""
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation set from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )

    print(
        f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(
    raw_path: str = "data/raw/cars_dataset_raw.csv",
    processed_train_path: str = "data/processed/X_train.csv",
    processed_val_path: str = "data/processed/X_val.csv",
    processed_test_path: str = "data/processed/X_test.csv",
    y_train_path: str = "data/processed/y_train.csv",
    y_val_path: str = "data/processed/y_val.csv",
    y_test_path: str = "data/processed/y_test.csv",
    save_dir: str = "data/processed",
) -> dict:
    """
    Full preprocessing pipeline.

    Returns:
        Dictionary with paths to saved files and preprocessing info
    """
    print("=" * 60)
    print("Starting preprocessing pipeline...")
    print("=" * 60)

    # Load data
    df = load_data(raw_path)

    # Clean column names
    df = clean_column_names(df)
    print(f"Columns after renaming: {df.columns.tolist()}")

    # Clean numeric columns
    df = clean_numeric_columns(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Feature engineering
    df = create_features(df)
    print(f"Features after engineering: {df.shape[1]}")

    # Encode categorical variables
    df_encoded, encoders = encode_categorical_variables(df, fit=True)
    print(f"Features after encoding: {df_encoded.shape[1]}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)

    # Save processed data
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_train_path, index=False)
    X_val.to_csv(processed_val_path, index=False)
    X_test.to_csv(processed_test_path, index=False)

    y_train.to_csv(y_train_path, index=False)
    y_val.to_csv(y_val_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    # Save preprocessing info
    preprocessing_info = {
        "original_shape": df.shape,
        "final_shape": df_encoded.shape,
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "paths": {
            "X_train": processed_train_path,
            "X_val": processed_val_path,
            "X_test": processed_test_path,
            "y_train": y_train_path,
            "y_val": y_val_path,
            "y_test": y_test_path,
        },
    }

    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print(f"Original shape: {preprocessing_info['original_shape']}")
    print(f"Final shape: {preprocessing_info['final_shape']}")
    print(f"Number of features: {preprocessing_info['n_features']}")
    print("=" * 60)

    return preprocessing_info


if __name__ == "__main__":
    # Run preprocessing pipeline
    info = preprocess_pipeline()
    print("\nPreprocessing info:", info)
