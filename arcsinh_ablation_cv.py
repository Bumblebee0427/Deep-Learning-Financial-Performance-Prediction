"""Raw versus arcsinh LightGBM targets for all nine targets on seed-2026 folds.

Both variants receive identical base features, preprocessing, folds, and model
hyperparameters. No test data and no weight/model selection in this script.
"""

from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd

from benchmark_models import matrices
from evaluation_framework import ID_COLUMN, ROOT, TARGETS, predictor_columns, score_predictions
from hard_target_oof import fit_lightgbm
from locked_recipe_cv import checked_cache, folds, load_train, save_cache, save_complete

OUTPUT = ROOT / "ablation_cv_2026"
CACHE = OUTPUT / "fold_cache"


def run() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    train = load_train()
    split = folds(train)
    features = predictor_columns(train)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    columns = [f"raw__{t}" for t in TARGETS] + [f"arcsinh__{t}" for t in TARGETS]
    output = np.full((len(train), len(columns)), np.nan, dtype=np.float64)
    fold_seconds = []
    for fold, (fit_idx, valid_idx) in enumerate(split):
        path = CACHE / f"fold{fold + 1}.parquet"
        ids = train.iloc[valid_idx][ID_COLUMN].to_numpy()
        values = checked_cache(path, valid_idx, ids, columns)
        if values is None:
            started = time.perf_counter()
            fit_rows, valid_rows = train.iloc[fit_idx], train.iloc[valid_idx]
            x_train, x_valid = matrices(fit_rows, valid_rows, features, numeric,
                                        categorical, scaled=False)
            y_train = fit_rows[list(TARGETS)].to_numpy(dtype=np.float64)
            raw = np.empty((len(valid_idx), len(TARGETS)), dtype=np.float64)
            arcsinh = np.empty_like(raw)
            for j, target in enumerate(TARGETS):
                raw[:, j] = fit_lightgbm(x_train, y_train[:, j], x_valid,
                                         fold=fold, transform=False)
                arcsinh[:, j] = fit_lightgbm(x_train, y_train[:, j], x_valid,
                                             fold=fold, transform=True)
                print(f"ablation fold {fold + 1}/5: {target}", flush=True)
            values = np.column_stack([raw, arcsinh])
            seconds = time.perf_counter() - started
            save_cache(path, valid_idx, ids, values, columns)
            del fit_rows, valid_rows, x_train, x_valid, y_train, raw, arcsinh
            gc.collect()
            print(f"ablation fold {fold + 1}/5: {seconds:.1f}s", flush=True)
        else:
            seconds = 0.0
            print(f"ablation fold {fold + 1}/5: loaded", flush=True)
        output[valid_idx] = values
        fold_seconds.append(seconds)
    if not np.isfinite(output).all():
        raise ValueError("Ablation OOF incomplete")
    save_complete(OUTPUT / "raw_oof.parquet", train, output[:, :len(TARGETS)], list(TARGETS))
    save_complete(OUTPUT / "arcsinh_oof.parquet", train, output[:, len(TARGETS):], list(TARGETS))
    truth = train[list(TARGETS)]
    raw_frame = pd.DataFrame(output[:, :len(TARGETS)], columns=TARGETS)
    arc_frame = pd.DataFrame(output[:, len(TARGETS):], columns=TARGETS)
    raw_scores, raw_mean = score_predictions(truth, raw_frame)
    arc_scores, arc_mean = score_predictions(truth, arc_frame)
    pd.DataFrame({
        "target": TARGETS,
        "raw_smape_percent": raw_scores.to_numpy(),
        "arcsinh_smape_percent": arc_scores.to_numpy(),
        "arcsinh_improvement_pp": raw_scores.to_numpy() - arc_scores.to_numpy(),
    }).to_csv(OUTPUT / "target_scores.csv", index=False)
    assignment = pd.read_parquet(ROOT / "locked_cv_2026" / "fold_assignments.parquet")
    fold_rows = []
    for fold in range(5):
        mask = assignment.fold.to_numpy() == fold
        for method, frame in (("raw", raw_frame), ("arcsinh", arc_frame)):
            scores, mean = score_predictions(truth.loc[mask], frame.loc[mask])
            fold_rows.append({"fold": fold + 1, "method": method,
                              "mean_smape_percent": mean, **scores.to_dict()})
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_scores.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps({
        "seed": 2026, "feature_set": "202 base predictors only",
        "raw_mean_smape_percent": raw_mean,
        "arcsinh_mean_smape_percent": arc_mean,
        "new_fold_seconds": fold_seconds,
    }, indent=2) + "\n")
    print(f"raw all-target OOF={raw_mean:.5f}%; arcsinh={arc_mean:.5f}%", flush=True)


if __name__ == "__main__":
    run()
