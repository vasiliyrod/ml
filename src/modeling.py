"""
Modeling module for Car Price Prediction.

This module handles:
- Baseline model training
- Multiple model experiments
- Hyperparameter tuning
- Model evaluation and comparison
- Ensemble methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import joblib
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_processed_data(
    train_path: str = "data/processed/X_train.csv",
    val_path: str = "data/processed/X_val.csv",
    test_path: str = "data/processed/X_test.csv",
    y_train_path: str = "data/processed/y_train.csv",
    y_val_path: str = "data/processed/y_val.csv",
    y_test_path: str = "data/processed/y_test.csv",
) -> tuple:
    """Load processed data from CSV files."""
    X_train = pd.read_csv(train_path)
    X_val = pd.read_csv(val_path)
    X_test = pd.read_csv(test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_val = pd.read_csv(y_val_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    print(f"Loaded data: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    return {"model": model_name, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2}


def train_baseline(X_train, y_train, X_val, y_val) -> dict:
    """Train baseline model (Linear Regression without feature engineering)."""
    print("\n" + "=" * 60)
    print("Training Baseline Model (Linear Regression)...")
    print("=" * 60)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_val_pred = model.predict(X_val)

    # Metrics
    metrics = calculate_metrics(y_val, y_val_pred, "Linear Regression (Baseline)")

    print("Validation Metrics:")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    return {"model": model, "metrics": metrics, "predictions": y_val_pred}


def train_multiple_models(X_train, y_train, X_val, y_val) -> dict:
    """Train multiple models and compare performance."""
    print("\n" + "=" * 60)
    print("Training Multiple Models...")
    print("=" * 60)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=10000),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            metrics = calculate_metrics(y_val, y_val_pred, name)
            results[name] = {
                "model": model,
                "metrics": metrics,
                "predictions": y_val_pred,
            }

            print(
                f"  RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE (%)']:.2f}%, R2: {metrics['R2']:.4f}"
            )
        except Exception as e:
            print(f"  Error training {name}: {str(e)}")
            results[name] = {"error": str(e)}

    return results


def hyperparameter_tuning(X_train, y_train, X_val, y_val) -> dict:
    """Perform hyperparameter tuning for best models."""
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning...")
    print("=" * 60)

    best_results = {}

    # Random Forest tuning
    print("\nTuning Random Forest...")
    rf_params = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 15},
        {"n_estimators": 200, "max_depth": 20},
        {"n_estimators": 100, "max_depth": None},
    ]

    best_rf_score = float("inf")
    best_rf_model = None
    best_rf_params = None

    for params in rf_params:
        rf = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        if rmse < best_rf_score:
            best_rf_score = rmse
            best_rf_model = rf
            best_rf_params = params

    metrics = calculate_metrics(
        y_val, best_rf_model.predict(X_val), f"Random Forest (Best: {best_rf_params})"
    )
    best_results["Random Forest Tuned"] = {
        "model": best_rf_model,
        "metrics": metrics,
        "params": best_rf_params,
    }
    print(f"  Best RF params: {best_rf_params}, RMSE: {best_rf_score:.2f}")

    # Gradient Boosting tuning
    print("\nTuning Gradient Boosting...")
    gb_params = [
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 200, "learning_rate": 0.01, "max_depth": 5},
    ]

    best_gb_score = float("inf")
    best_gb_model = None
    best_gb_params = None

    for params in gb_params:
        gb = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        if rmse < best_gb_score:
            best_gb_score = rmse
            best_gb_model = gb
            best_gb_params = params

    metrics = calculate_metrics(
        y_val,
        best_gb_model.predict(X_val),
        f"Gradient Boosting (Best: {best_gb_params})",
    )
    best_results["Gradient Boosting Tuned"] = {
        "model": best_gb_model,
        "metrics": metrics,
        "params": best_gb_params,
    }
    print(f"  Best GB params: {best_gb_params}, RMSE: {best_gb_score:.2f}")

    return best_results


def create_ensemble(X_train, y_train, X_val, y_val, models_dict: dict) -> dict:
    """Create ensemble model from best individual models."""
    print("\n" + "=" * 60)
    print("Creating Ensemble Model...")
    print("=" * 60)

    # Get predictions from top models
    predictions = []
    model_names = []

    # Select top 3 models based on validation performance
    sorted_models = sorted(
        [(name, info) for name, info in models_dict.items() if "metrics" in info],
        key=lambda x: x[1]["metrics"]["RMSE"],
    )[:3]

    for name, info in sorted_models:
        if "predictions" in info:
            predictions.append(info["predictions"])
            model_names.append(name)
            print(f"  Including {name} in ensemble")

    if len(predictions) < 2:
        print("  Not enough models for ensemble")
        return {}

    # Simple averaging ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    metrics = calculate_metrics(y_val, ensemble_pred, "Ensemble (Average)")

    print("\nEnsemble Metrics:")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    return {
        "model": {"type": "ensemble", "models": model_names},
        "metrics": metrics,
        "predictions": ensemble_pred,
    }


def evaluate_on_test(X_test, y_test, models_dict: dict) -> dict:
    """Evaluate best models on test set."""
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set...")
    print("=" * 60)

    test_results = {}

    for name, info in models_dict.items():
        if "model" in info and "predictions" not in info:
            # Need to predict on test set
            model = info["model"]
            if isinstance(model, dict) and model.get("type") == "ensemble":
                continue  # Skip ensemble for now
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, name)
            test_results[name] = metrics
            print(
                f"{name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE (%)']:.2f}%, R2={metrics['R2']:.4f}"
            )
        elif "predictions" in info:
            # Already have predictions (for ensemble or validation)
            # Retrain and predict on test
            if "model" in info and not isinstance(info["model"], dict):
                model = info["model"]
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, name)
                test_results[name] = metrics
                print(
                    f"{name}: RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE (%)']:.2f}%, R2={metrics['R2']:.4f}"
                )

    return test_results


def save_best_model(
    model_info, model_name: str, save_path: str = "models/best_model.pkl"
):
    """Save the best model to disk."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if "model" in model_info and not isinstance(model_info["model"], dict):
        joblib.dump(model_info["model"], save_path)
        print(f"Model saved to {save_path}")
    else:
        print("Cannot save ensemble or invalid model")


def create_experiment_table(all_results: dict) -> pd.DataFrame:
    """Create a DataFrame with all experiment results."""
    rows = []

    for name, info in all_results.items():
        if "metrics" in info:
            row = info["metrics"].copy()
            if "params" in info:
                row["params"] = str(info["params"])
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("RMSE")

    return df


def modeling_pipeline() -> dict:
    """Full modeling pipeline."""
    print("=" * 60)
    print("Starting Modeling Pipeline...")
    print("=" * 60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

    # Train baseline
    baseline_result = train_baseline(X_train, y_train, X_val, y_val)

    # Train multiple models
    all_results = train_multiple_models(X_train, y_train, X_val, y_val)
    all_results["Baseline"] = baseline_result

    # Hyperparameter tuning
    tuned_results = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    all_results.update(tuned_results)

    # Create ensemble
    ensemble_result = create_ensemble(X_train, y_train, X_val, y_val, all_results)
    if ensemble_result:
        all_results["Ensemble"] = ensemble_result

    # Create experiment table
    experiment_df = create_experiment_table(all_results)
    print("\n" + "=" * 60)
    print("Experiment Results Summary:")
    print("=" * 60)
    print(experiment_df.to_string(index=False))

    # Save experiment results
    experiment_df.to_csv("reports/experiment_results.csv", index=False)
    print("\nExperiment results saved to reports/experiment_results.csv")

    # Find best model
    if not experiment_df.empty:
        best_model_name = experiment_df.iloc[0]["model"]
        best_model_info = all_results.get(best_model_name, {})

        print(f"\n{'=' * 60}")
        print(f"Best Model: {best_model_name}")
        print(f"RMSE: {experiment_df.iloc[0]['RMSE']:.2f}")
        print(f"MAPE: {experiment_df.iloc[0]['MAPE (%)']:.2f}%")
        print(f"R2: {experiment_df.iloc[0]['R2']:.4f}")
        print(f"{'=' * 60}")

        # Save best model
        save_best_model(best_model_info, best_model_name)

    # Evaluate on test set
    test_results = evaluate_on_test(X_test, y_test, all_results)

    return {
        "all_results": all_results,
        "experiment_df": experiment_df,
        "test_results": test_results,
    }


if __name__ == "__main__":
    results = modeling_pipeline()
