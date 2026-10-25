
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_preprocessing import (
    load_and_validate_data,
    prepare_features,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    nonzero_targets = y_test != 0
    mape = (
        np.mean(
            np.abs(
                (y_test[nonzero_targets] - predictions[nonzero_targets])
                / y_test[nonzero_targets]
            )
        ) * 100
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": float(mape),
    }


def train_models():
    df = load_and_validate_data()
    X, y = prepare_features(df)

    # Keep a held-out test set for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    preprocessor = build_preprocessor()

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    # Cross-validation is performed on training data only.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", estimator),
        ])

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
        )

        pipeline.fit(X_train, y_train)
        test_metrics = evaluate_model(pipeline, X_test, y_test)

        results.append({
            "Model": name,
            **test_metrics,
            "CV_MAE_mean": float(
                -cv_results["test_mae"].mean()
            ),
            "CV_MAE_std": float(
                cv_results["test_mae"].std()
            ),
            "CV_R2_mean": float(
                cv_results["test_r2"].mean()
            ),
        })

        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(DOCS_DIR / "model_comparison.csv", index=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Select the best baseline using cross-validated MAE.
    best_name = min(
        fitted_models,
        key=lambda name: next(
            row["CV_MAE_mean"]
            for row in results
            if row["Model"] == name
        ),
    )

    print(f"\nBest baseline by CV MAE: {best_name}")

    # Tune a Random Forest using training data only.
    tuning_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("regressor", RandomForestRegressor(
            random_state=42,
            n_jobs=1,
        )),
    ])

    parameter_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10],
        "regressor__min_samples_leaf": [1, 2],
    }

    search = GridSearchCV(
        tuning_pipeline,
        parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)

    print("\n========== TUNED RANDOM FOREST ==========")
    print("Best parameters:", search.best_params_)
    for metric, value in tuned_metrics.items():
        print(f"{metric}: {value:,.4f}")

    # Save the selected model and metadata.
    # The test set is not used to choose between the baseline
    # and tuned model; selection uses training CV results.
    tuned_cv_mae = -search.best_score_
    baseline_cv_mae = min(
        row["CV_MAE_mean"] for row in results
    )

    if tuned_cv_mae <= baseline_cv_mae:
        selected_model = tuned_model
        selected_name = "Tuned Random Forest"
    else:
        selected_model = fitted_models[best_name]
        selected_name = best_name

    model_path = MODEL_DIR / "house_price_model_v1.joblib"
    joblib.dump(selected_model, model_path)

    metadata = {
        "model_name": selected_name,
        "version": "1.0",
        "target": "Price",
        "features": X.columns.tolist(),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "selection_metric": "5-fold cross-validation MAE",
        "selected_cv_mae": float(
            min(tuned_cv_mae, baseline_cv_mae)
        ),
        "final_test_metrics": tuned_metrics
        if selected_name == "Tuned Random Forest"
        else next(
            row for row in results if row["Model"] == selected_name
        ),
        "best_random_forest_parameters": search.best_params_,
    }

    with open(MODEL_DIR / "model_metadata_v1.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nSelected model: {selected_name}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    train_models()
