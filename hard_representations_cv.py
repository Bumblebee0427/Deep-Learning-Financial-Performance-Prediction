"""OOF hard-target representation experiments on the locked seed-2026 folds.

All supervised transforms and preprocessing are fitted within each fold.
Revenue and operating-income reconstruction uses aligned locked OOF values.
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from benchmark_models import TREE_THREADS, matrices
from evaluation_framework import ID_COLUMN, ROOT, TARGETS, predictor_columns, score_predictions
from hard_target_oof import lag_features
from locked_recipe_cv import (OUTPUT as LOCKED, checked_cache, folds, load_train,
                              read_aligned, save_cache, save_complete)

OUTPUT = ROOT / f"representation_cv_{os.environ.get('REPRESENTATION_CV_SEED', '2026')}"
CACHE = OUTPUT / "fold_cache"
HARD_TARGETS = ("Q0_OPERATING_INCOME", "Q0_EBITDA")
METHODS = ("direct_arcsinh", "margin", "sign_magnitude", "residual")


def scale_of(values: np.ndarray) -> float:
    scale = float(np.median(np.abs(values)))
    return scale if scale > 0 else 1.0


def estimator(fold: int, *, classifier: bool = False):
    cls = LGBMClassifier if classifier else LGBMRegressor
    return cls(n_estimators=120, learning_rate=0.08, num_leaves=31,
               min_child_samples=40, colsample_bytree=0.8,
               n_jobs=TREE_THREADS, random_state=42 + fold,
               deterministic=True, force_col_wise=True, verbosity=-1)


def transformed_fit_predict(x_fit, values, x_valid, fold):
    scale = scale_of(values)
    model = estimator(fold)
    model.fit(x_fit, np.arcsinh(values / scale))
    return np.sinh(model.predict(x_valid)) * scale


def score_one(truth: pd.DataFrame, target: str, values: np.ndarray) -> float:
    comparison = truth.copy()
    comparison[target] = values
    scores, _ = score_predictions(truth, comparison)
    return float(scores[target])


def run_target(target: str, data: pd.DataFrame, split, locked: pd.DataFrame) -> None:
    engineered = lag_features(data, target)
    augmented = pd.concat([data, engineered], axis=1)
    features = predictor_columns(data) + list(engineered.columns)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(augmented[c])]
    categorical = [c for c in features if c not in numeric]
    active = list(METHODS if target == "Q0_EBITDA" else METHODS[:-1])
    output = np.full((len(data), len(active)), np.nan, dtype=np.float64)

    direct = read_aligned(LOCKED / f"hard_{target}_oof.parquet", data, [target])[target].to_numpy()
    output[:, 0] = direct
    for fold, (fit_idx, valid_idx) in enumerate(split):
        path = CACHE / f"{target}_fold{fold + 1}.parquet"
        ids = data.iloc[valid_idx][ID_COLUMN].to_numpy()
        predicted = checked_cache(path, valid_idx, ids, active[1:])
        if predicted is None:
            started = time.perf_counter()
            fit_rows, valid_rows = augmented.iloc[fit_idx], augmented.iloc[valid_idx]
            x_fit, x_valid = matrices(fit_rows, valid_rows, features, numeric,
                                      categorical, scaled=False)
            y = fit_rows[target].to_numpy(dtype=np.float64)
            train_revenue = fit_rows["Q0_REVENUES"].to_numpy(dtype=np.float64)
            if (train_revenue <= 0).any():
                raise ValueError("Revenue is nonpositive; margin definition needs review")
            margin = y / train_revenue
            margin_pred = transformed_fit_predict(x_fit, margin, x_valid, fold)
            revenue_oof = locked.iloc[valid_idx]["Q0_REVENUES"].to_numpy(dtype=np.float64)
            margin_reconstructed = margin_pred * revenue_oof

            classifier = estimator(fold, classifier=True)
            classifier.fit(x_fit, (y > 0).astype(np.int8))
            positive = classifier.predict_proba(x_valid)[:, 1] >= 0.5
            magnitude = transformed_fit_predict(x_fit, np.abs(y), x_valid, fold)
            sign_magnitude = np.where(positive, magnitude, -magnitude)
            parts = [margin_reconstructed, sign_magnitude]

            if target == "Q0_EBITDA":
                residual = y - fit_rows["Q0_OPERATING_INCOME"].to_numpy(dtype=np.float64)
                residual_pred = transformed_fit_predict(x_fit, residual, x_valid, fold)
                oi_oof = locked.iloc[valid_idx]["Q0_OPERATING_INCOME"].to_numpy(dtype=np.float64)
                parts.append(oi_oof + residual_pred)
            predicted = np.column_stack(parts)
            if not np.isfinite(predicted).all():
                raise ValueError("Non-finite hard-target prediction")
            save_cache(path, valid_idx, ids, predicted, active[1:])
            print(f"{target} representations fold {fold + 1}/5: "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
            del fit_rows, valid_rows, x_fit, x_valid, classifier
            gc.collect()
        else:
            print(f"{target} representations fold {fold + 1}/5: loaded", flush=True)
        output[valid_idx, 1:] = predicted

    save_complete(OUTPUT / f"{target}_oof.parquet", data, output, active)
    truth = data[list(TARGETS)]
    assignments = pd.read_parquet(LOCKED / "fold_assignments.parquet").fold.to_numpy()
    rows, fold_rows = [], []
    actual = data[target].to_numpy(dtype=np.float64)
    for j, method in enumerate(active):
        rows.append({"target": target, "method": method,
                     "smape_percent": score_one(truth, target, output[:, j]),
                     "sign_accuracy_percent": 100 * np.mean(np.sign(actual) == np.sign(output[:, j]))})
        for fold in range(5):
            mask = assignments == fold
            fold_rows.append({"target": target, "method": method, "fold": fold + 1,
                              "smape_percent": score_one(truth.loc[mask], target,
                                                          output[mask, j])})
    pd.DataFrame(rows).to_csv(OUTPUT / f"{target}_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / f"{target}_fold_scores.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    data = load_train()
    split = folds(data)
    locked = read_aligned(LOCKED / "locked_final_oof.parquet", data, list(TARGETS))
    for target in HARD_TARGETS:
        run_target(target, data, split, locked)


if __name__ == "__main__":
    main()
