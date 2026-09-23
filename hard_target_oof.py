"""Controlled 5-fold LightGBM experiments for three difficult targets.

Both variants use the same engineered lag features and folds. The only change
between them is raw versus train-fold arcsinh target values. No test file used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex-financial")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from benchmark_models import TREE_THREADS, matrices
from evaluation_framework import ID_COLUMN, RANDOM_STATE, ROOT, TARGETS, predictor_columns, score_predictions

OUTPUT = ROOT / "oof_outputs"
CACHE = OUTPUT / "fold_cache"
HARD_TARGETS = ("Q0_OPERATING_INCOME", "Q0_EBITDA", "Q0_GROSS_PROFIT")
N_SPLITS = 5
CONFIG = {"n_estimators": 120, "learning_rate": 0.08, "num_leaves": 31,
          "min_child_samples": 40, "colsample_bytree": 0.8, "threads": TREE_THREADS,
          "n_splits": 5, "random_state": RANDOM_STATE, "lag_feature_version": 1,
          "preprocessing": "fit-fold median imputation + one-hot categories"}
TAG = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def lag_features(train: pd.DataFrame, target: str) -> pd.DataFrame:
    """Row-wise Q1-Q10 features; positive slope means growth toward Q1."""
    suffix = target.removeprefix("Q0_")
    values = np.column_stack([train[f"Q{k}_{suffix}"].to_numpy(dtype=np.float64)
                              for k in range(1, 11)])
    if not np.isfinite(values).all():
        raise ValueError(f"Lag values for {target} are not finite")
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
    }, index=train.index)


def fit_lightgbm(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
                 *, fold: int, transform: bool) -> np.ndarray:
    from lightgbm import LGBMRegressor

    if transform:
        scale = float(np.median(np.abs(y_train)))
        if scale <= 0:
            scale = 1.0
        fitted_target = np.arcsinh(y_train / scale)
    else:
        scale = 1.0
        fitted_target = y_train
    model = LGBMRegressor(
        n_estimators=120, learning_rate=0.08, num_leaves=31,
        min_child_samples=40, colsample_bytree=0.8,
        n_jobs=TREE_THREADS, random_state=RANDOM_STATE + fold,
        deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(x_train, fitted_target)
    predicted = model.predict(x_valid)
    del model
    if transform:
        predicted = np.sinh(predicted) * scale
    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions from target inverse transformation")
    return np.asarray(predicted, dtype=np.float64)


def target_score(truth: pd.DataFrame, target: str, prediction: np.ndarray) -> float:
    surrogate = truth.copy()
    surrogate[target] = prediction
    per_target, _ = score_predictions(truth, surrogate)
    return float(per_target[target])


def run_target(target: str, *, force: bool = False) -> None:
    if target not in HARD_TARGETS:
        raise ValueError(target)
    CACHE.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv", low_memory=False)
    base = predictor_columns(train)
    engineered = lag_features(train, target)
    train = pd.concat([train, engineered], axis=1)
    features = base + list(engineered.columns)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    if ID_COLUMN in features or set(TARGETS) & set(features):
        raise AssertionError("Id or Q0 target entered predictors")
    output = np.full((len(train), 2), np.nan, dtype=np.float64)
    fold_times = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(train)):
        path = CACHE / f"hard_{target}_{TAG}_fold{fold + 1}.parquet"
        if path.exists() and not force:
            cached = pd.read_parquet(path)
            if not np.array_equal(cached.row_index.to_numpy(), valid_idx) or not np.array_equal(
                cached[ID_COLUMN].to_numpy(), train.iloc[valid_idx][ID_COLUMN].to_numpy()
            ):
                raise ValueError("Hard-target fold cache alignment mismatch")
            values = cached[["raw_lag", "arcsinh_lag"]].to_numpy(dtype=np.float64)
            seconds = float(cached.fold_seconds.iloc[0])
            print(f"{target} fold {fold + 1}/5: loaded checkpoint", flush=True)
        else:
            started = time.perf_counter()
            fit_rows, valid_rows = train.iloc[fit_idx], train.iloc[valid_idx]
            x_train, x_valid = matrices(
                fit_rows, valid_rows, features, numeric, categorical, scaled=False
            )
            y_train = fit_rows[target].to_numpy(dtype=np.float64)
            raw = fit_lightgbm(x_train, y_train, x_valid, fold=fold, transform=False)
            transformed = fit_lightgbm(x_train, y_train, x_valid, fold=fold, transform=True)
            values = np.column_stack([raw, transformed])
            seconds = time.perf_counter() - started
            cached = pd.DataFrame({"row_index": valid_idx,
                                   ID_COLUMN: train.iloc[valid_idx][ID_COLUMN].to_numpy(),
                                   "raw_lag": raw, "arcsinh_lag": transformed,
                                   "fold_seconds": seconds})
            cached.to_parquet(path, index=False, compression="zstd")
            print(f"{target} fold {fold + 1}/5: {seconds:.1f}s", flush=True)
            del fit_rows, valid_rows, x_train, x_valid, y_train, cached
            gc.collect()
        output[valid_idx] = values
        fold_times.append(seconds)
    if not np.isfinite(output).all():
        raise ValueError("Hard-target OOF matrix incomplete")
    aligned = pd.DataFrame({"row_index": np.arange(len(train)), ID_COLUMN: train[ID_COLUMN],
                            "raw_lag": output[:, 0], "arcsinh_lag": output[:, 1]})
    aligned.to_parquet(OUTPUT / f"hard_{target}_oof.parquet", index=False, compression="zstd")
    truth = train[list(TARGETS)]
    actual = train[target].to_numpy(dtype=np.float64)
    metrics = {"target": target, "config": CONFIG, "engineered_features": list(engineered.columns),
               "fold_seconds": fold_times}
    for j, method in enumerate(("raw_lag", "arcsinh_lag")):
        metrics[method] = {
            "smape_percent": target_score(truth, target, output[:, j]),
            "sign_accuracy_percent": 100.0 * float(np.mean(np.sign(actual) == np.sign(output[:, j]))),
            "prediction_min": float(output[:, j].min()),
            "prediction_max": float(output[:, j].max()),
        }
    (OUTPUT / f"hard_{target}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{target}: raw+lags={metrics['raw_lag']['smape_percent']:.4f}%, "
          f"arcsinh+lags={metrics['arcsinh_lag']['smape_percent']:.4f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", choices=HARD_TARGETS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.all:
        for target in HARD_TARGETS:
            command = [sys.executable, str(Path(__file__).resolve()), "--target", target]
            if args.force:
                command.append("--force")
            subprocess.run(command, check=True)
    else:
        run_target(args.target, force=args.force)


if __name__ == "__main__":
    main()
