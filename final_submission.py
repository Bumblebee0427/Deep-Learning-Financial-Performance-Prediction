"""Fit the frozen candidate recipe on all training rows and write the submission.

Usage: python final_submission.py
No CV, search, tuning, or test-based model selection occurs here.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-financial")

import numpy as np
import pandas as pd

from benchmark_models import matrices
from candidate_recipe_cv import arcsinh_model, recipe
from evaluation_framework import ID_COLUMN, ROOT, TARGETS, predictor_columns
from hard_representations_cv import estimator as representation_estimator
from hard_representations_cv import scale_of, transformed_fit_predict
from hard_target_oof import lag_features
from lightgbm_small_search import CONFIGS
from locked_recipe_cv import RECIPE_SHA256, apply_accounting_recipe

TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_individual.csv"
SELECTION_PATH = ROOT / "candidate_cv_314159" / "selected_substitutions.json"
FROZEN_RECIPE_PATH = ROOT / "locked_cv_2026" / "target_recipes_frozen.csv"
SOURCE_RECIPE_PATH = ROOT / "oof_outputs" / "target_recipes.csv"
SUBMISSION_PATH = ROOT / "final_submission.csv"
DIAGNOSTICS_PATH = ROOT / "final_submission_diagnostics.json"

# These are the eight substitutions that passed the seed-314159 check.
# Q0_REVENUES is computed from direct gross profit and cost predictions.
DIRECT_METHODS = {
    "Q0_TOTAL_ASSETS": "search_large_leaf",
    "Q0_TOTAL_LIABILITIES": "search_large_leaf",
    "Q0_TOTAL_STOCKHOLDERS_EQUITY": "search_large_leaf",
    "Q0_GROSS_PROFIT": "search_l1",
    "Q0_COST_OF_REVENUES": "search_l1",
    "Q0_OPERATING_INCOME": "sign_magnitude",
    "Q0_OPERATING_EXPENSES": "search_large_leaf",
    "Q0_EBITDA": "sign_magnitude",
}
LAG_TARGETS = {"Q0_GROSS_PROFIT", "Q0_OPERATING_INCOME", "Q0_EBITDA"}
EXPECTED_COLUMNS = [ID_COLUMN, *TARGETS]


def check_fixed_recipe() -> pd.DataFrame:
    selected = json.loads(SELECTION_PATH.read_text())
    if selected.get("substitutions") != DIRECT_METHODS:
        raise ValueError("Selected candidate methods differ from the fixed final recipe")
    common = {"learning_rate": 0.08, "n_estimators": 120}
    expected_configs = {
        "large_leaf": {**common, "num_leaves": 63, "min_child_samples": 20,
                       "objective": "regression"},
        "l1": {**common, "num_leaves": 31, "min_child_samples": 40,
               "objective": "regression_l1"},
    }
    for name, expected_config in expected_configs.items():
        if CONFIGS[name] != expected_config:
            raise ValueError(f"Validated LightGBM configuration changed: {name}")
    sign_params = representation_estimator(0, classifier=True).get_params()
    for name, value in {"n_estimators": 120, "learning_rate": 0.08,
                        "num_leaves": 31, "min_child_samples": 40,
                        "colsample_bytree": 0.8, "random_state": 42}.items():
        if sign_params[name] != value:
            raise ValueError(f"Validated sign classifier parameter changed: {name}")
    if (hashlib.sha256(FROZEN_RECIPE_PATH.read_bytes()).hexdigest() != RECIPE_SHA256 or
        hashlib.sha256(SOURCE_RECIPE_PATH.read_bytes()).hexdigest() != RECIPE_SHA256):
        raise ValueError("Frozen accounting recipe hash changed")
    weights = recipe()
    expected = {
        "Q0_TOTAL_ASSETS": ("direct", 1.0, False),
        "Q0_TOTAL_LIABILITIES": ("assets_minus_equity", 0.5, False),
        "Q0_TOTAL_STOCKHOLDERS_EQUITY": ("assets_minus_liabilities", 0.0, False),
        "Q0_GROSS_PROFIT": ("direct", 1.0, False),
        "Q0_COST_OF_REVENUES": ("direct", 1.0, False),
        "Q0_REVENUES": ("gross_profit_plus_cost", 0.0, True),
        "Q0_OPERATING_INCOME": ("gross_profit_minus_expenses", 0.5, False),
        "Q0_OPERATING_EXPENSES": ("gross_profit_minus_operating_income", 0.25, True),
        "Q0_EBITDA": ("direct", 1.0, False),
    }
    for target, (formula, direct_weight, guard) in expected.items():
        row = weights.loc[target]
        if (row.derived_formula != formula or
            float(row.direct_weight) != direct_weight or
            (str(row.nonnegative_guard).lower() == "true") != guard):
            raise ValueError(f"Accounting recipe changed for {target}")
    return weights


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False, dtype={ID_COLUMN: "string"})
    if len(train) != 100_000 or len(test) != 50_000:
        raise ValueError("Unexpected train or test row count")
    if list(train.columns).count(ID_COLUMN) != 1 or list(test.columns).count(ID_COLUMN) != 1:
        raise ValueError("Id must occur exactly once in each input")
    features = predictor_columns(train)
    if ID_COLUMN in features or set(TARGETS).intersection(features):
        raise AssertionError("Id or Q0 target entered predictors")
    if [column for column in test.columns if column != ID_COLUMN] != features:
        raise ValueError("Test predictor names/order differ from training")
    if test[ID_COLUMN].isna().any() or not test[ID_COLUMN].is_unique:
        raise ValueError("Test Id contains missing or duplicate values")
    if not np.isfinite(train[list(TARGETS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Training targets contain non-finite values")
    return train, test, features


def feature_matrices(train: pd.DataFrame, test: pd.DataFrame,
                     base_features: list[str], target: str):
    """Reuse the exact CV lag constructor and fold preprocessing on full train."""
    if target in LAG_TARGETS:
        train_lags = lag_features(train, target)
        test_lags = lag_features(test, target)
        train_rows = pd.concat([train, train_lags], axis=1)
        test_rows = pd.concat([test, test_lags], axis=1)
        features = base_features + list(train_lags.columns)
        if list(train_lags.columns) != list(test_lags.columns):
            raise ValueError(f"Lag feature mismatch for {target}")
    else:
        train_rows, test_rows, features = train, test, base_features
    if ID_COLUMN in features or set(TARGETS).intersection(features):
        raise AssertionError("Id or target entered final feature matrix")
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train_rows[c])]
    categorical = [c for c in features if c not in numeric]
    return matrices(train_rows, test_rows, features, numeric, categorical, scaled=False)


def fit_direct(train: pd.DataFrame, test: pd.DataFrame,
               features: list[str]) -> pd.DataFrame:
    direct = pd.DataFrame(index=test.index, columns=TARGETS, dtype=np.float64)
    base_x_train, base_x_test = feature_matrices(
        train, test, features, "Q0_TOTAL_ASSETS"
    )
    for target, method in DIRECT_METHODS.items():
        if target in LAG_TARGETS:
            x_train, x_test = feature_matrices(train, test, features, target)
        else:
            x_train, x_test = base_x_train, base_x_test
        y = train[target].to_numpy(dtype=np.float64)
        if method == "sign_magnitude":
            classifier = representation_estimator(0, classifier=True)
            classifier.fit(x_train, (y > 0).astype(np.int8))
            positive = classifier.predict_proba(x_test)[:, 1] >= 0.5
            magnitude = transformed_fit_predict(x_train, np.abs(y), x_test, 0)
            prediction = np.where(positive, magnitude, -magnitude)
            del classifier
        else:
            scale = scale_of(y)  # Full training target only, as in each CV fold.
            model = arcsinh_model(0, method)
            model.fit(x_train, np.arcsinh(y / scale))
            prediction = np.sinh(model.predict(x_test)) * scale
            del model
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite direct predictions for {target}")
        direct[target] = prediction
        print(f"trained {target}: {method}", flush=True)
        if target in LAG_TARGETS:
            del x_train, x_test
            gc.collect()

    # The frozen recipe gives revenues zero direct weight. This value only
    # completes the direct frame required by the shared accounting helper.
    direct["Q0_REVENUES"] = direct["Q0_GROSS_PROFIT"] + direct["Q0_COST_OF_REVENUES"]
    return direct.loc[:, list(TARGETS)]


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Diagnostics require finite values")
    return {
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "number_negative": int((values < 0).sum()),
    }


def diagnostics(train: pd.DataFrame, submission: pd.DataFrame) -> dict:
    targets = {}
    for target in TARGETS:
        prediction = distribution(submission[target].to_numpy())
        training = distribution(train[target].to_numpy())
        targets[target] = {
            "prediction": prediction,
            "training_target": training,
            "comparison": {
                "median_difference": prediction["median"] - training["median"],
                "mean_difference": prediction["mean"] - training["mean"],
                "p01_difference": prediction["p01"] - training["p01"],
                "p99_difference": prediction["p99"] - training["p99"],
                "negative_rate_difference_percentage_points": 100.0 * (
                    prediction["number_negative"] / len(submission)
                    - training["number_negative"] / len(train)
                ),
            },
        }
    return {
        "train_rows": len(train),
        "test_rows": len(submission),
        "output_columns": EXPECTED_COLUMNS,
        "frozen_accounting_recipe_sha256": RECIPE_SHA256,
        "fixed_methods": DIRECT_METHODS,
        "targets": targets,
    }


def validate_submission(submission: pd.DataFrame, test_ids: pd.Series) -> None:
    if submission.shape != (50_000, 10) or list(submission.columns) != EXPECTED_COLUMNS:
        raise ValueError("Submission shape or rubric column order is wrong")
    if not submission[ID_COLUMN].equals(test_ids):
        raise ValueError("Test Id values/order changed")
    if submission[ID_COLUMN].isna().any() or not submission[ID_COLUMN].is_unique:
        raise ValueError("Submission Id contains missing or duplicate values")
    predictions = submission.loc[:, list(TARGETS)].to_numpy(dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("Submission contains NaN or infinity")
    if (submission["Q0_REVENUES"] < 0).any():
        raise ValueError("Revenues contain negative values")
    if (submission["Q0_OPERATING_EXPENSES"] < 0).any():
        raise ValueError("Operating expenses contain negative values")


def main() -> None:
    weights = check_fixed_recipe()
    train, test, features = load_data()
    direct = fit_direct(train, test, features)
    final = apply_accounting_recipe(direct, weights)
    submission = pd.concat([test[[ID_COLUMN]], final.loc[:, list(TARGETS)]], axis=1)
    validate_submission(submission, test[ID_COLUMN])
    report = diagnostics(train, submission)

    # Validate the on-disk CSV before replacing the final artifact.
    temporary = Path(str(SUBMISSION_PATH) + ".tmp")
    try:
        submission.to_csv(temporary, index=False)
        round_trip = pd.read_csv(temporary, low_memory=False, dtype={ID_COLUMN: "string"})
        validate_submission(round_trip, test[ID_COLUMN])
        os.replace(temporary, SUBMISSION_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()
    DIAGNOSTICS_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"saved {SUBMISSION_PATH} ({len(submission):,} rows)", flush=True)
    print(f"saved {DIAGNOSTICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
