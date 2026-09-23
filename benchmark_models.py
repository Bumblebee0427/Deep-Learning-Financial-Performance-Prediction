"""Train-only, single-split regression benchmark. Never reads test_individual.csv.

Run all models in separate processes (limits peak memory):
    python benchmark_models.py --all
Or run one model, then rebuild the comparison:
    python benchmark_models.py --model ridge
    python benchmark_models.py --summarize
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from evaluation_framework import (
    ID_COLUMN,
    RANDOM_STATE,
    ROOT,
    TARGETS,
    mean_baseline,
    predictor_columns,
    score_predictions,
    split_training_data,
)

OUTPUT = ROOT / "benchmark_outputs"
PREDICTIONS = OUTPUT / "validation_predictions"
MODELS = ("mean", "ridge", "random_forest", "extra_trees", "hist_gradient_boosting", "lightgbm", "neural_network")
TREE_THREADS = 4


def data_split() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    train = pd.read_csv(ROOT / "train.csv", low_memory=False)
    features = predictor_columns(train)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    train_part, valid_part = split_training_data(train)
    expected = ROOT / "audit_outputs" / "split_ids.csv"
    if expected.exists():
        previous = pd.read_csv(expected)
        valid_ids = previous.loc[previous["split"] == "validation", ID_COLUMN].to_numpy()
        if len(valid_ids) != len(valid_part) or set(valid_ids) != set(valid_part[ID_COLUMN]):
            raise ValueError("Validation IDs differ from the existing audit split")
    if ID_COLUMN in features or set(TARGETS).intersection(features):
        raise AssertionError("Id or target leaked into predictors")
    return train_part, valid_part, features, numeric, categorical


def preprocessor(numeric: list[str], categorical: list[str], *, scaled: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scaled:
        numeric_steps += [
            ("signed_log", FunctionTransformer(np.arcsinh, validate=False)),
            ("scale", StandardScaler()),
        ]
    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ]
    return ColumnTransformer(
        [("numeric", Pipeline(numeric_steps), numeric),
         ("categorical", Pipeline(categorical_steps), categorical)],
        sparse_threshold=0.0,
    )


def matrices(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    *,
    scaled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    processor = preprocessor(numeric, categorical, scaled=scaled)
    x_train = processor.fit_transform(train_part[features]).astype(np.float32, copy=False)
    x_valid = processor.transform(valid_part[features]).astype(np.float32, copy=False)
    if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
        raise ValueError("Preprocessed features contain non-finite values")
    return x_train, x_valid


def train_neural_network(
    x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Small 2-layer network; transform and scaling statistics come from train only."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    fit_start = time.perf_counter()
    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(TREE_THREADS)
    scale = np.median(np.abs(y_train), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    transformed = np.arcsinh(y_train / scale)
    center = transformed.mean(axis=0)
    spread = transformed.std(axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    z_train = ((transformed - center) / spread).astype(np.float32)

    network = nn.Sequential(
        nn.Linear(x_train.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, len(TARGETS)),
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(z_train)),
        batch_size=2048, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    network.train()
    for epoch in range(12):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = network(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
        print(f"neural_network epoch {epoch + 1:02d}/12 loss={total_loss / len(x_train):.5f}", flush=True)
    fit_seconds = time.perf_counter() - fit_start
    prediction_start = time.perf_counter()
    network.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_valid), 4096):
            chunks.append(network(torch.from_numpy(x_valid[start:start + 4096])).numpy())
    z_pred = np.vstack(chunks).astype(np.float64)
    values = np.sinh(z_pred * spread + center) * scale
    return values, fit_seconds, time.perf_counter() - prediction_start


def fit_model(
    name: str,
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, float, float, dict]:
    """Training time includes train-fold preprocessing and fit, not CSV loading."""
    y_train = train_part[list(TARGETS)].to_numpy(dtype=np.float64)
    start = time.perf_counter()
    config: dict = {}
    if name == "mean":
        pred = mean_baseline(train_part, valid_part)
        training_seconds = time.perf_counter() - start
        return pred, training_seconds, 0.0, {"estimator": "train-fold target mean"}

    scaled = name in ("ridge", "neural_network")
    x_train, x_valid = matrices(
        train_part, valid_part, features, numeric, categorical, scaled=scaled
    )
    preprocessing_seconds = time.perf_counter() - start
    config["preprocessing"] = "median imputation, one-hot categories" + (
        ", arcsinh and standard scaling of numeric features" if scaled else ""
    )
    config["encoded_feature_count"] = int(x_train.shape[1])

    if name == "neural_network":
        config.update({"architecture": [128, 64], "epochs": 12, "batch_size": 2048,
                       "loss": "SmoothL1", "target_transform": "arcsinh(y / median_abs_train)"})
        values, fit_seconds, predict_seconds = train_neural_network(x_train, y_train, x_valid)
        training_seconds = preprocessing_seconds + fit_seconds
    else:
        if name == "ridge":
            model = Ridge(alpha=100.0, solver="lsqr")
            config.update({"alpha": 100.0, "solver": "lsqr"})
        elif name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=48, max_depth=14, min_samples_leaf=8,
                max_features=0.7, n_jobs=TREE_THREADS, random_state=RANDOM_STATE,
            )
            config.update({"n_estimators": 48, "max_depth": 14,
                           "min_samples_leaf": 8, "max_features": 0.7})
        elif name == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=48, max_depth=14, min_samples_leaf=8,
                max_features=0.7, n_jobs=TREE_THREADS, random_state=RANDOM_STATE,
            )
            config.update({"n_estimators": 48, "max_depth": 14,
                           "min_samples_leaf": 8, "max_features": 0.7})
        elif name == "hist_gradient_boosting":
            model = None
            config.update({"max_iter": 80, "max_leaf_nodes": 31,
                           "l2_regularization": 1.0, "target_models": 9})
        elif name == "lightgbm":
            from lightgbm import LGBMRegressor

            model = None
            config.update({"n_estimators": 120, "learning_rate": 0.08,
                           "num_leaves": 31, "min_child_samples": 40,
                           "target_models": 9})
        else:
            raise ValueError(f"Unknown model: {name}")

        if name in ("hist_gradient_boosting", "lightgbm"):
            fitted = []
            for j, target in enumerate(TARGETS):
                if name == "hist_gradient_boosting":
                    estimator = HistGradientBoostingRegressor(
                        max_iter=80, max_leaf_nodes=31, max_bins=128,
                        l2_regularization=1.0, early_stopping=False,
                        random_state=RANDOM_STATE,
                    )
                else:
                    estimator = LGBMRegressor(
                        n_estimators=120, learning_rate=0.08, num_leaves=31,
                        min_child_samples=40, colsample_bytree=0.8,
                        n_jobs=TREE_THREADS, random_state=RANDOM_STATE,
                        deterministic=True, force_col_wise=True, verbosity=-1,
                    )
                estimator.fit(x_train, y_train[:, j])
                fitted.append(estimator)
                print(f"{name}: fitted {target} ({j + 1}/9)", flush=True)
            training_seconds = time.perf_counter() - start
            predict_start = time.perf_counter()
            values = np.column_stack([est.predict(x_valid) for est in fitted])
            predict_seconds = time.perf_counter() - predict_start
        else:
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - start
            predict_start = time.perf_counter()
            values = model.predict(x_valid)
            predict_seconds = time.perf_counter() - predict_start

    prediction = pd.DataFrame(np.asarray(values, dtype=np.float64),
                              index=valid_part.index, columns=TARGETS)
    return prediction, training_seconds, predict_seconds, config


def diagnostics(train_part: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    """Descriptive range checks; no validation-driven clipping or fitting."""
    result = {}
    for target in TARGETS:
        truth = train_part[target].to_numpy(dtype=np.float64)
        pred = prediction[target].to_numpy(dtype=np.float64)
        scale = float(np.quantile(np.abs(truth), 0.99))
        result[target] = {
            "prediction_min": float(pred.min()),
            "prediction_max": float(pred.max()),
            "prediction_median": float(np.median(pred)),
            "outside_training_range_count": int(((pred < truth.min()) | (pred > truth.max())).sum()),
            "above_10x_training_abs_p99_count": int((np.abs(pred) > 10 * scale).sum()),
            "negative_prediction_count": int((pred < 0).sum()),
        }
    return result


def run_model(name: str) -> None:
    if name not in MODELS:
        raise ValueError(name)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    train_part, valid_part, features, numeric, categorical = data_split()
    prediction, train_seconds, pred_seconds, config = fit_model(
        name, train_part, valid_part, features, numeric, categorical
    )
    y_valid = valid_part[list(TARGETS)]
    per_target, average = score_predictions(y_valid, prediction)

    # Id is recorded solely as an alignment key, never used as a predictor.
    aligned = pd.DataFrame({"row_index": valid_part.index.to_numpy(),
                            ID_COLUMN: valid_part[ID_COLUMN].to_numpy()}, index=valid_part.index)
    if not (OUTPUT / "validation_truth.csv").exists():
        aligned.join(y_valid).to_csv(OUTPUT / "validation_truth.csv", index=False,
                                     float_format="%.17g")
    else:
        previous = pd.read_csv(OUTPUT / "validation_truth.csv", usecols=["row_index", ID_COLUMN])
        if not np.array_equal(previous["row_index"], aligned["row_index"]) or not np.array_equal(
            previous[ID_COLUMN], aligned[ID_COLUMN]
        ):
            raise ValueError("Saved validation truth uses a different row order")
    aligned.join(prediction).to_csv(PREDICTIONS / f"{name}.csv", index=False,
                                    float_format="%.17g")
    metrics = {
        "model": name,
        "training_seconds": train_seconds,
        "prediction_seconds": pred_seconds,
        "mean_smape_percent": average,
        "target_smape_percent": {t: float(per_target[t]) for t in TARGETS},
        "validation_rows": len(valid_part),
        "random_state": RANDOM_STATE,
        "config": config,
        "range_diagnostics": diagnostics(train_part, prediction),
    }
    (OUTPUT / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{name}: mean sMAPE={average:.4f}%, training={train_seconds:.1f}s, "
          f"prediction={pred_seconds:.1f}s", flush=True)


def summarize() -> None:
    files = [OUTPUT / f"{name}_metrics.json" for name in MODELS]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model results: {missing}")
    metrics = [json.loads(path.read_text()) for path in files]
    baseline = next(m["mean_smape_percent"] for m in metrics if m["model"] == "mean")
    rows = []
    for m in metrics:
        row = {
            "model": m["model"],
            "mean_smape_percent": m["mean_smape_percent"],
            "improvement_vs_mean_percent": 100.0 * (baseline - m["mean_smape_percent"]) / baseline,
            "training_seconds": m["training_seconds"],
            "prediction_seconds": m["prediction_seconds"],
        }
        row.update(m["target_smape_percent"])
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values("mean_smape_percent", kind="stable")
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    print(comparison[["model", "mean_smape_percent", "improvement_vs_mean_percent",
                      "training_seconds"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--all", action="store_true")
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.all:
        for name in MODELS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--model", name], check=True)
        summarize()
    elif args.summarize:
        summarize()
    else:
        run_model(args.model)


if __name__ == "__main__":
    main()
