"""Training-only data audit and reproducible naive baseline for individual prediction.

Usage: python evaluation_framework.py
Outputs are written to audit_outputs/. No prediction file is generated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "audit_outputs"
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
ID_COLUMN = "Id"
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20


def predictor_columns(train: pd.DataFrame) -> list[str]:
    """Use the training schema only; exclude labels and Id by construction."""
    required = set(TARGETS) | {ID_COLUMN}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Missing required training columns: {sorted(missing)}")
    return [name for name in train.columns if name not in required]


def smape_by_target(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    """Rubric: 100/n * sum(|y-p| / (0.5*(|y|+|p|))) per target.

    The rubric leaves 0/0 undefined; use the standard convention of zero error
    when both actual and prediction are zero. All other terms follow the formula.
    """
    if list(y_true.columns) != list(TARGETS) or list(y_pred.columns) != list(TARGETS):
        raise ValueError("Both frames must contain the nine targets in rubric order")
    if y_true.shape != y_pred.shape or not y_true.index.equals(y_pred.index):
        raise ValueError("Truth and prediction must have identical shape and row index")
    actual = y_true.to_numpy(dtype=np.float64)
    predicted = y_pred.to_numpy(dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("sMAPE requires finite, non-missing targets and predictions")
    denominator = 0.5 * (np.abs(actual) + np.abs(predicted))
    terms = np.divide(
        np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return pd.Series(100.0 * terms.mean(axis=0), index=TARGETS, name="smape_percent")


def score_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> tuple[pd.Series, float]:
    """Shared scoring entry point for future models."""
    per_target = smape_by_target(y_true, y_pred)
    return per_target, float(per_target.mean())


def split_training_data(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows, rather than columns, are partitioned; no test data is involved."""
    return train_test_split(
        train, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE, shuffle=True
    )


def mean_baseline(train_part: pd.DataFrame, validation_part: pd.DataFrame) -> pd.DataFrame:
    """Fit each constant from training labels only."""
    means = train_part.loc[:, TARGETS].mean(axis=0)
    return pd.DataFrame(
        np.broadcast_to(means.to_numpy(dtype=np.float64), (len(validation_part), len(TARGETS))).copy(),
        index=validation_part.index,
        columns=TARGETS,
    )


def column_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Full column and missing-value inventory, including literal placeholders."""
    rows = []
    for name in frame.columns:
        series = frame[name]
        missing = int(series.isna().sum())
        nonfinite = (
            int((~np.isfinite(series.to_numpy(dtype=np.float64)) & series.notna().to_numpy()).sum())
            if pd.api.types.is_numeric_dtype(series)
            else 0
        )
        placeholders = (
            int(series.isin(["Missing", "Unknown"]).sum())
            if pd.api.types.is_object_dtype(series)
            else 0
        )
        rows.append(
            {
                "column": name,
                "dtype": str(series.dtype),
                "kind": (
                    "target" if name in TARGETS else "id" if name == ID_COLUMN
                    else "categorical" if not pd.api.types.is_numeric_dtype(series)
                    else "numeric"
                ),
                "missing_count": missing,
                "missing_pct": 100.0 * missing / len(frame),
                "nonfinite_count": nonfinite,
                "placeholder_count": placeholders,
                "unique_nonmissing": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def target_audit(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in TARGETS:
        values = pd.to_numeric(train[name], errors="raise")
        finite = values[np.isfinite(values)]
        if finite.empty:
            raise ValueError(f"Target {name} has no finite values")
        rows.append(
            {
                "target": name,
                "count": int(values.notna().sum()),
                "missing_count": int(values.isna().sum()),
                "nonfinite_count": int((~np.isfinite(values) & values.notna()).sum()),
                "negative_count": int((finite < 0).sum()),
                "zero_count": int((finite == 0).sum()),
                "min": float(finite.min()),
                "p01": float(finite.quantile(0.01)),
                "median": float(finite.median()),
                "mean": float(finite.mean()),
                "p99": float(finite.quantile(0.99)),
                "max": float(finite.max()),
            }
        )
    return pd.DataFrame(rows)


def fiscal_flag_audit(train: pd.DataFrame) -> pd.DataFrame:
    """Check nominal binary quarter-end indicators for synthetic fractional values."""
    rows = []
    for quarter in range(11):
        name = f"Q{quarter}_fiscal_year_end"
        values = train[name].to_numpy(dtype=np.float64)
        is_zero = np.isclose(values, 0)
        is_one = np.isclose(values, 1)
        rows.append({
            "column": name,
            "near_zero_count": int(is_zero.sum()),
            "near_one_count": int(is_one.sum()),
            "nonbinary_count": int((~is_zero & ~is_one).sum()),
        })
    return pd.DataFrame(rows)


def same_quarter_metadata_audit(train: pd.DataFrame) -> pd.DataFrame:
    """Train-only diagnostic for metadata that resembles Q0 labels."""
    pairs = [("totalRevenue", "Q0_REVENUES"), ("ebitda", "Q0_EBITDA")]
    return pd.DataFrame([
        {
            "predictor": predictor,
            "target": target,
            "pearson_correlation": float(train[predictor].corr(train[target])),
            "exact_match_count": int(train[predictor].eq(train[target]).sum()),
        }
        for predictor, target in pairs
    ])


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv", low_memory=False)
    features = predictor_columns(train)
    if len(features) != len(set(features)) or not train.columns.is_unique:
        raise ValueError("Duplicate training column names")

    train_columns = column_audit(train)
    target_stats = target_audit(train)
    train_columns.to_csv(OUTPUT / "train_columns.csv", index=False)
    target_stats.to_csv(OUTPUT / "target_distributions.csv", index=False)
    fiscal_flags = fiscal_flag_audit(train)
    fiscal_flags.to_csv(OUTPUT / "fiscal_flags.csv", index=False)
    same_quarter_metadata_audit(train).to_csv(OUTPUT / "same_quarter_metadata.csv", index=False)

    train_part, validation_part = split_training_data(train)
    # These variables document the exact model interface for later estimators.
    X_train, X_valid = train_part[features], validation_part[features]
    y_train, y_valid = train_part[list(TARGETS)], validation_part[list(TARGETS)]
    if y_train.isna().any().any() or y_valid.isna().any().any():
        raise ValueError("Targets contain missing values; decide a label policy before scoring")
    predictions = mean_baseline(train_part, validation_part)
    scores, average = score_predictions(y_valid, predictions)
    split_labels = pd.Series("train", index=train.index, name="split")
    split_labels.loc[validation_part.index] = "validation"
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], "split": split_labels}).to_csv(
        OUTPUT / "split_ids.csv", index=False
    )
    baseline = pd.DataFrame(
        {"target": TARGETS, "training_mean": y_train.mean().to_numpy(),
         "validation_smape_percent": scores.to_numpy()}
    )
    baseline.to_csv(OUTPUT / "baseline_scores.csv", index=False)

    # Read the held-out file only after the training-only split and baseline are fixed.
    # It is used exclusively for schema/missingness checks, never for fitting/tuning.
    test = pd.read_csv(ROOT / "test_individual.csv", low_memory=False)
    test_columns = column_audit(test)
    test_columns.to_csv(OUTPUT / "test_columns.csv", index=False)
    test_features = [name for name in test.columns if name != ID_COLUMN]
    missing_from_test = sorted(set(features) - set(test_features))
    extra_in_test = sorted(set(test_features) - set(features))
    same_order = features == test_features

    categorical = [name for name in features if not pd.api.types.is_numeric_dtype(train[name])]
    numeric = [name for name in features if pd.api.types.is_numeric_dtype(train[name])]
    fiscal_columns = [name for name in features if name.endswith("_fiscal_year_end")]
    historical = [name for name in features if name.startswith(tuple(f"Q{i}_" for i in range(1, 11)))]
    metadata = [name for name in features if name not in historical + fiscal_columns]
    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_columns": list(TARGETS),
        "predictor_count": len(features),
        "numeric_predictors": numeric,
        "categorical_predictors": categorical,
        "train_test_predictors_match": not missing_from_test and not extra_in_test,
        "train_test_predictor_order_matches": same_order,
        "missing_from_test": missing_from_test,
        "extra_in_test": extra_in_test,
        "id_train_unique": bool(train[ID_COLUMN].is_unique),
        "id_test_unique": bool(test[ID_COLUMN].is_unique),
        "id_overlap_count": int(len(set(train[ID_COLUMN]) & set(test[ID_COLUMN]))),
        "duplicate_train_predictor_rows": int(train[features].duplicated().sum()),
        "duplicate_train_historical_rows": int(train[historical].duplicated().sum()),
        "train_missing_cells": int(train.isna().sum().sum()),
        "test_missing_cells": int(test.isna().sum().sum()),
        "train_nonfinite_numeric_cells": int(train_columns.nonfinite_count.sum()),
        "test_nonfinite_numeric_cells": int(test_columns.nonfinite_count.sum()),
        "train_nonbinary_fiscal_cells": int(fiscal_flags.nonbinary_count.sum()),
        "validation_fraction": VALIDATION_FRACTION,
        "random_state": RANDOM_STATE,
        "train_split_rows": len(X_train),
        "validation_split_rows": len(X_valid),
        "baseline_mean_smape_percent": average,
        "q0_metadata_columns_for_leakage_review": metadata,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "train_shape": summary["train_shape"],
        "test_shape": summary["test_shape"],
        "predictor_count": summary["predictor_count"],
        "categorical_predictors": categorical,
        "predictors_match": summary["train_test_predictors_match"],
        "baseline_mean_smape_percent": average,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
