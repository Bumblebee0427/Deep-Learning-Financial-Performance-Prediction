"""Five-fold OOF predictions for the three selected tree families.

Only train.csv is read. Run `python oof_models.py --all`, or one model at a time
with `--model random_forest|lightgbm|hist_gradient_boosting`.
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

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

from benchmark_models import TREE_THREADS, matrices
from evaluation_framework import ID_COLUMN, RANDOM_STATE, ROOT, TARGETS, predictor_columns, score_predictions

OUTPUT = ROOT / "oof_outputs"
CACHE = OUTPUT / "fold_cache"
MODELS = ("random_forest", "lightgbm", "hist_gradient_boosting")
N_SPLITS = 5
PREPROCESS_VERSION = 1


def model_config(name: str) -> dict:
    shared = {"folds": N_SPLITS, "shuffle": True, "random_state": RANDOM_STATE,
              "preprocess_version": PREPROCESS_VERSION,
              "preprocessing": "train-fold median imputation + one-hot categories"}
    if name == "random_forest":
        return shared | {"n_estimators": 48, "max_depth": 14, "min_samples_leaf": 8,
                         "max_features": 0.7, "n_jobs": TREE_THREADS}
    if name == "lightgbm":
        return shared | {"n_estimators": 120, "learning_rate": 0.08,
                         "num_leaves": 31, "min_child_samples": 40,
                         "colsample_bytree": 0.8, "n_jobs": TREE_THREADS}
    if name == "hist_gradient_boosting":
        return shared | {"max_iter": 80, "max_leaf_nodes": 31, "max_bins": 128,
                         "l2_regularization": 1.0, "early_stopping": False}
    raise ValueError(name)


def configuration_tag(name: str) -> str:
    payload = json.dumps(model_config(name), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def train_predict_fold(
    name: str, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray,
    fold: int,
) -> np.ndarray:
    if name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=48, max_depth=14, min_samples_leaf=8,
            max_features=0.7, n_jobs=TREE_THREADS, random_state=RANDOM_STATE + fold,
        )
        estimator.fit(x_train, y_train)
        values = estimator.predict(x_valid)
        del estimator
        return np.asarray(values, dtype=np.float64)

    values = np.empty((len(x_valid), len(TARGETS)), dtype=np.float64)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
    for j, target in enumerate(TARGETS):
        if name == "lightgbm":
            estimator = LGBMRegressor(
                n_estimators=120, learning_rate=0.08, num_leaves=31,
                min_child_samples=40, colsample_bytree=0.8,
                n_jobs=TREE_THREADS, random_state=RANDOM_STATE + fold,
                deterministic=True, force_col_wise=True, verbosity=-1,
            )
        else:
            estimator = HistGradientBoostingRegressor(
                max_iter=80, max_leaf_nodes=31, max_bins=128,
                l2_regularization=1.0, early_stopping=False,
                random_state=RANDOM_STATE + fold,
            )
        estimator.fit(x_train, y_train[:, j])
        values[:, j] = estimator.predict(x_valid)
        del estimator
        print(f"{name} fold {fold + 1}/{N_SPLITS}: {target} done", flush=True)
    return values


def run_model(name: str, *, force: bool = False) -> None:
    if name not in MODELS:
        raise ValueError(name)
    CACHE.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv", low_memory=False)
    features = predictor_columns(train)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    if ID_COLUMN in features or set(TARGETS) & set(features):
        raise AssertionError("Id or targets entered the feature set")
    truth = train[list(TARGETS)]
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_by_row = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(splitter.split(train)):
        fold_by_row[valid_idx] = fold
    if (fold_by_row < 0).any():
        raise AssertionError("A training row has no validation fold")
    assignment = pd.DataFrame({"row_index": np.arange(len(train)),
                               ID_COLUMN: train[ID_COLUMN].to_numpy(),
                               "fold": fold_by_row})
    assignment_path = OUTPUT / "fold_assignments.parquet"
    if assignment_path.exists():
        existing = pd.read_parquet(assignment_path)
        if not existing.equals(assignment):
            raise ValueError("Existing fold assignments do not match")
    else:
        assignment.to_parquet(assignment_path, index=False, compression="zstd")
    output = np.full((len(train), len(TARGETS)), np.nan, dtype=np.float64)
    fold_times = []
    tag = configuration_tag(name)

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(train)):
        fold_path = CACHE / f"{name}_{tag}_fold{fold + 1}.parquet"
        if fold_path.exists() and not force:
            cached = pd.read_parquet(fold_path)
            if not np.array_equal(cached["row_index"].to_numpy(), valid_idx) or not np.array_equal(
                cached[ID_COLUMN].to_numpy(), train.iloc[valid_idx][ID_COLUMN].to_numpy()
            ):
                raise ValueError(f"Cached fold alignment mismatch: {fold_path}")
            values = cached[list(TARGETS)].to_numpy(dtype=np.float64)
            seconds = float(cached["fold_seconds"].iloc[0])
            print(f"{name} fold {fold + 1}/{N_SPLITS}: loaded checkpoint", flush=True)
        else:
            started = time.perf_counter()
            fit_rows, valid_rows = train.iloc[fit_idx], train.iloc[valid_idx]
            x_train, x_valid = matrices(
                fit_rows, valid_rows, features, numeric, categorical, scaled=False
            )
            y_train = fit_rows[list(TARGETS)].to_numpy(dtype=np.float64)
            values = train_predict_fold(name, x_train, y_train, x_valid, fold)
            seconds = time.perf_counter() - started
            if values.shape != (len(valid_idx), len(TARGETS)) or not np.isfinite(values).all():
                raise ValueError(f"Invalid OOF predictions for {name}, fold {fold + 1}")
            cached = pd.DataFrame(values, columns=TARGETS)
            cached.insert(0, ID_COLUMN, train.iloc[valid_idx][ID_COLUMN].to_numpy())
            cached.insert(0, "row_index", valid_idx)
            cached["fold_seconds"] = seconds
            cached.to_parquet(fold_path, index=False, compression="zstd")
            print(f"{name} fold {fold + 1}/{N_SPLITS}: {seconds:.1f}s", flush=True)
            del fit_rows, valid_rows, x_train, x_valid, y_train, cached
            gc.collect()
        output[valid_idx] = values
        fold_times.append(seconds)

    if not np.isfinite(output).all():
        raise ValueError("OOF matrix is incomplete or non-finite")
    prediction = pd.DataFrame(output, columns=TARGETS)
    per_target, mean = score_predictions(truth, prediction)
    aligned = prediction.copy()
    aligned.insert(0, ID_COLUMN, train[ID_COLUMN].to_numpy())
    aligned.insert(0, "row_index", np.arange(len(train)))
    aligned.to_parquet(OUTPUT / f"{name}_oof.parquet", index=False, compression="zstd")
    metrics = {
        "model": name, "n_rows": len(train), "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE, "config": model_config(name),
        "fold_seconds": fold_times, "total_fold_seconds": float(sum(fold_times)),
        "mean_smape_percent": mean,
        "target_smape_percent": {t: float(per_target[t]) for t in TARGETS},
    }
    (OUTPUT / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{name} OOF mean sMAPE={mean:.5f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="Refit even if fold checkpoints exist")
    args = parser.parse_args()
    if args.all:
        for name in MODELS:
            command = [sys.executable, str(Path(__file__).resolve()), "--model", name]
            if args.force:
                command.append("--force")
            subprocess.run(command, check=True)
    else:
        run_model(args.model, force=args.force)


if __name__ == "__main__":
    main()
