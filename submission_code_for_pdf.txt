"""Final individual prediction pipeline; run with `python submission_code.py`.

Requires train.csv and test_individual.csv beside this script. The recipe is
fixed. All fitting uses the 100,000 training rows; test labels are never used.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Configuration
ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_individual.csv"
OUTPUT_PATH = ROOT / "submission_code_output.csv"
ID = "Id"
TARGETS = (
    "Q0_TOTAL_ASSETS",
    "Q0_TOTAL_LIABILITIES",
    "Q0_TOTAL_STOCKHOLDERS_EQUITY",
    "Q0_GROSS_PROFIT",
    "Q0_COST_OF_REVENUES",
    "Q0_REVENUES",
    "Q0_OPERATING_INCOME",
    "Q0_OPERATING_EXPENSES",
    "Q0_EBITDA",
)
LAG_TARGETS = {"Q0_GROSS_PROFIT", "Q0_OPERATING_INCOME", "Q0_EBITDA"}
DIRECT_METHODS = {
    "Q0_TOTAL_ASSETS": "large_leaf",
    "Q0_TOTAL_LIABILITIES": "large_leaf",
    "Q0_TOTAL_STOCKHOLDERS_EQUITY": "large_leaf",
    "Q0_GROSS_PROFIT": "l1",
    "Q0_COST_OF_REVENUES": "l1",
    "Q0_OPERATING_INCOME": "sign_magnitude",
    "Q0_OPERATING_EXPENSES": "large_leaf",
    "Q0_EBITDA": "sign_magnitude",
}
THREADS = 4
SEED = 42


# Data loading
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False, dtype={ID: "string"})
    if len(train) != 100_000 or len(test) != 50_000:
        raise ValueError("Unexpected train or test row count")
    if any(target not in train.columns for target in TARGETS):
        raise ValueError("Training targets are missing")
    if ID not in train.columns or ID not in test.columns:
        raise ValueError("Id is missing")
    predictors = [name for name in train.columns if name not in set(TARGETS) | {ID}]
    if ID in predictors or set(predictors) & set(TARGETS):
        raise AssertionError("Id or a Q0 target entered the predictors")
    if predictors != [name for name in test.columns if name != ID]:
        raise ValueError("Train/test predictor columns or their order differ")
    if test[ID].isna().any() or not test[ID].is_unique:
        raise ValueError("Test Id contains missing or duplicate values")
    if not np.isfinite(train[list(TARGETS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Training targets contain non-finite values")
    return train, test, predictors


# Feature preprocessing
def make_matrices(train_rows: pd.DataFrame, test_rows: pd.DataFrame,
                  predictors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit median imputation and one-hot categories on training rows only."""
    if ID in predictors or set(predictors) & set(TARGETS):
        raise AssertionError("Id or target entered the feature matrix")
    numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(train_rows[c])]
    categorical = [c for c in predictors if c not in numeric]
    processor = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False, dtype=np.float32)),
            ]), categorical),
        ],
        sparse_threshold=0.0,
    )
    x_train = processor.fit_transform(train_rows[predictors]).astype(np.float32, copy=False)
    x_test = processor.transform(test_rows[predictors]).astype(np.float32, copy=False)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Preprocessed features contain non-finite values")
    return x_train, x_test


# Lag feature engineering
def lag_features(rows: pd.DataFrame, target: str) -> pd.DataFrame:
    """The exact eleven Q1-Q10 features used by the validated pipeline."""
    suffix = target.removeprefix("Q0_")
    values = np.column_stack([rows[f"Q{k}_{suffix}"].to_numpy(dtype=np.float64)
                              for k in range(1, 11)])
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite lag values for {target}")
    recent = values[:, :4]
    weights4 = np.arange(3, -1, -1, dtype=np.float64) - 1.5
    weights10 = np.arange(9, -1, -1, dtype=np.float64) - 4.5
    stem = target.lower()
    return pd.DataFrame({
        f"{stem}_lag_q1": values[:, 0],
        f"{stem}_q1_minus_q2": values[:, 0] - values[:, 1],
        f"{stem}_q1_sign": np.sign(values[:, 0]),
        f"{stem}_q2_sign": np.sign(values[:, 1]),
        f"{stem}_q1_q2_same_sign": np.sign(values[:, 0]) * np.sign(values[:, 1]),
        f"{stem}_q1_q4_mean": recent.mean(axis=1),
        f"{stem}_q1_q4_median": np.median(recent, axis=1),
        f"{stem}_q1_q4_std": recent.std(axis=1),
        f"{stem}_q1_q4_slope_recent": recent @ weights4 / np.dot(weights4, weights4),
        f"{stem}_q1_q10_mean": values.mean(axis=1),
        f"{stem}_q1_q10_slope_recent": values @ weights10 / np.dot(weights10, weights10),
    }, index=rows.index)


def matrices_for_target(train: pd.DataFrame, test: pd.DataFrame,
                        base_predictors: list[str], target: str):
    if target not in LAG_TARGETS:
        return make_matrices(train, test, base_predictors)
    train_lags = lag_features(train, target)
    test_lags = lag_features(test, target)
    if list(train_lags.columns) != list(test_lags.columns):
        raise ValueError(f"Lag feature names differ for {target}")
    train_rows = pd.concat([train, train_lags], axis=1)
    test_rows = pd.concat([test, test_lags], axis=1)
    return make_matrices(train_rows, test_rows,
                         base_predictors + list(train_lags.columns))


# Target transformation
def target_scale(values: np.ndarray) -> float:
    scale = float(np.median(np.abs(values)))
    return scale if scale > 0 else 1.0


def arcsinh_prediction(model: LGBMRegressor, x_train: np.ndarray,
                       y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    scale = target_scale(y_train)  # Only the full training target determines scale.
    model.fit(x_train, np.arcsinh(y_train / scale))
    return np.sinh(model.predict(x_test)) * scale


# LightGBM helpers
def regression_model(method: str) -> LGBMRegressor:
    if method == "large_leaf":
        config = {"num_leaves": 63, "min_child_samples": 20,
                  "objective": "regression"}
    elif method == "l1":
        config = {"num_leaves": 31, "min_child_samples": 40,
                  "objective": "regression_l1"}
    else:
        raise ValueError(f"Unknown direct regression method: {method}")
    return LGBMRegressor(
        **config, learning_rate=0.08, n_estimators=120,
        colsample_bytree=0.8, n_jobs=THREADS, random_state=SEED,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )


def sign_magnitude_model(*, classifier: bool):
    cls = LGBMClassifier if classifier else LGBMRegressor
    return cls(
        n_estimators=120, learning_rate=0.08, num_leaves=31,
        min_child_samples=40, colsample_bytree=0.8,
        n_jobs=THREADS, random_state=SEED,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )


# Sign + magnitude models
def predict_signed_magnitude(x_train: np.ndarray, y_train: np.ndarray,
                             x_test: np.ndarray) -> np.ndarray:
    classifier = sign_magnitude_model(classifier=True)
    classifier.fit(x_train, (y_train > 0).astype(np.int8))
    positive = classifier.predict_proba(x_test)[:, 1] >= 0.5
    magnitude = arcsinh_prediction(
        sign_magnitude_model(classifier=False), x_train, np.abs(y_train), x_test
    )
    return np.where(positive, magnitude, -magnitude)


# Train final direct models
def train_direct_models(train: pd.DataFrame, test: pd.DataFrame,
                        predictors: list[str]) -> pd.DataFrame:
    direct = pd.DataFrame(index=test.index, columns=TARGETS, dtype=np.float64)
    base_x_train, base_x_test = matrices_for_target(
        train, test, predictors, "Q0_TOTAL_ASSETS"
    )
    for target, method in DIRECT_METHODS.items():
        if target in LAG_TARGETS:
            x_train, x_test = matrices_for_target(train, test, predictors, target)
        else:
            x_train, x_test = base_x_train, base_x_test
        y_train = train[target].to_numpy(dtype=np.float64)
        if method == "sign_magnitude":
            prediction = predict_signed_magnitude(x_train, y_train, x_test)
        else:
            prediction = arcsinh_prediction(
                regression_model(method), x_train, y_train, x_test
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite direct prediction for {target}")
        direct[target] = prediction
        print(f"trained {target}", flush=True)
        if target in LAG_TARGETS:
            del x_train, x_test
            gc.collect()
    return direct


# Accounting reconciliation
def reconcile_direct_predictions(direct: pd.DataFrame) -> pd.DataFrame:
    """Every formula below reads only direct predictions, never final values."""
    d_a = direct["Q0_TOTAL_ASSETS"]
    d_l = direct["Q0_TOTAL_LIABILITIES"]
    d_e = direct["Q0_TOTAL_STOCKHOLDERS_EQUITY"]
    d_gp = direct["Q0_GROSS_PROFIT"]
    d_c = direct["Q0_COST_OF_REVENUES"]
    d_oi = direct["Q0_OPERATING_INCOME"]
    d_oe = direct["Q0_OPERATING_EXPENSES"]
    d_ebitda = direct["Q0_EBITDA"]
    final = pd.DataFrame(index=direct.index)
    final["Q0_TOTAL_ASSETS"] = d_a
    final["Q0_TOTAL_LIABILITIES"] = 0.50 * d_l + 0.50 * (d_a - d_e)
    final["Q0_TOTAL_STOCKHOLDERS_EQUITY"] = d_a - d_l
    final["Q0_GROSS_PROFIT"] = d_gp
    final["Q0_COST_OF_REVENUES"] = d_c
    final["Q0_REVENUES"] = np.maximum(0.0, d_gp + d_c)
    final["Q0_OPERATING_INCOME"] = 0.50 * d_oi + 0.50 * (d_gp - d_oe)
    final["Q0_OPERATING_EXPENSES"] = np.maximum(
        0.0, 0.25 * d_oe + 0.75 * (d_gp - d_oi)
    )
    final["Q0_EBITDA"] = d_ebitda
    return final.loc[:, list(TARGETS)]


# Submission validation
def validate_submission(submission: pd.DataFrame, original_ids: pd.Series) -> None:
    if submission.shape != (50_000, 10) or list(submission.columns) != [ID, *TARGETS]:
        raise ValueError("Submission shape or column order differs from rubric")
    if not submission[ID].equals(original_ids):
        raise ValueError("Test Id values or order changed")
    if submission[ID].isna().any() or not submission[ID].is_unique:
        raise ValueError("Submission Id has missing or duplicate values")
    if not np.isfinite(submission[list(TARGETS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Submission contains NaN or infinite predictions")
    if (submission["Q0_REVENUES"] < 0).any():
        raise ValueError("Revenue predictions are negative")
    if (submission["Q0_OPERATING_EXPENSES"] < 0).any():
        raise ValueError("Operating expense predictions are negative")


# Save predictions
def main() -> None:
    train, test, predictors = load_data()
    direct = train_direct_models(train, test, predictors)
    final = reconcile_direct_predictions(direct)
    submission = pd.concat([test[[ID]], final], axis=1)
    validate_submission(submission, test[ID])
    submission.to_csv(OUTPUT_PATH, index=False)
    saved = pd.read_csv(OUTPUT_PATH, low_memory=False, dtype={ID: "string"})
    validate_submission(saved, test[ID])
    print(f"saved {OUTPUT_PATH} ({len(saved):,} rows)", flush=True)


if __name__ == "__main__":
    main()
