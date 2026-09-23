"""Controlled arcsinh LightGBM search on development folds (seed 2026).

The search chooses configurations only on these folds. A separate seed is
reserved for evaluating the chosen candidates after selection is frozen.
"""

from __future__ import annotations

import gc
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from benchmark_models import TREE_THREADS, matrices
from evaluation_framework import ID_COLUMN, ROOT, TARGETS, predictor_columns, score_predictions
from hard_target_oof import HARD_TARGETS, lag_features
from locked_recipe_cv import checked_cache, folds, load_train, save_cache, save_complete

OUTPUT = ROOT / "search_cv_2026"
CACHE = OUTPUT / "fold_cache"
ABLATION = ROOT / "ablation_cv_2026"

# Each row alters a controlled aspect of the existing 120-tree configuration.
CONFIGS = {
    "baseline": {"num_leaves": 31, "min_child_samples": 40,
                 "learning_rate": 0.08, "n_estimators": 120, "objective": "regression"},
    "small_leaf": {"num_leaves": 15, "min_child_samples": 80,
                   "learning_rate": 0.08, "n_estimators": 120, "objective": "regression"},
    "large_leaf": {"num_leaves": 63, "min_child_samples": 20,
                   "learning_rate": 0.08, "n_estimators": 120, "objective": "regression"},
    "slow": {"num_leaves": 31, "min_child_samples": 40,
             "learning_rate": 0.05, "n_estimators": 180, "objective": "regression"},
    "l1": {"num_leaves": 31, "min_child_samples": 40,
           "learning_rate": 0.08, "n_estimators": 120, "objective": "regression_l1"},
}


def score_one(data: pd.DataFrame, target: str, values: np.ndarray) -> float:
    truth = data[list(TARGETS)]
    predicted = truth.copy()
    predicted[target] = values
    return float(score_predictions(truth, predicted)[0][target])


def eligible_targets() -> list[str]:
    scores = pd.read_csv(ABLATION / "target_scores.csv").set_index("target")
    fold_scores = pd.read_csv(ABLATION / "fold_scores.csv")
    selected = []
    for target in TARGETS:
        pivot = fold_scores.pivot(index="fold", columns="method", values=target)
        fold_wins = int((pivot["arcsinh"] < pivot["raw"]).sum())
        improvement = float(scores.loc[target, "arcsinh_improvement_pp"])
        if improvement >= 0.15 and fold_wins >= 3:
            selected.append(target)
    return selected


def run_target(target: str, data: pd.DataFrame, split) -> dict:
    engineered = lag_features(data, target) if target in HARD_TARGETS else None
    full = pd.concat([data, engineered], axis=1) if engineered is not None else data
    features = predictor_columns(data) + (list(engineered.columns) if engineered is not None else [])
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(full[c])]
    categorical = [c for c in features if c not in numeric]
    names = list(CONFIGS)
    output = np.full((len(data), len(names)), np.nan, dtype=np.float64)
    assignment = np.full(len(data), -1, dtype=np.int8)
    for fold, (fit_idx, valid_idx) in enumerate(split):
        assignment[valid_idx] = fold
        path = CACHE / f"{target}_fold{fold + 1}.parquet"
        ids = data.iloc[valid_idx][ID_COLUMN].to_numpy()
        values = checked_cache(path, valid_idx, ids, names)
        if values is None:
            started = time.perf_counter()
            fit_rows, valid_rows = full.iloc[fit_idx], full.iloc[valid_idx]
            x_fit, x_valid = matrices(fit_rows, valid_rows, features, numeric,
                                      categorical, scaled=False)
            y = fit_rows[target].to_numpy(dtype=np.float64)
            scale = float(np.median(np.abs(y))) or 1.0
            transformed = np.arcsinh(y / scale)
            values = np.empty((len(valid_idx), len(names)), dtype=np.float64)
            for j, name in enumerate(names):
                model = LGBMRegressor(**CONFIGS[name], colsample_bytree=0.8,
                                      n_jobs=TREE_THREADS, random_state=42 + fold,
                                      deterministic=True, force_col_wise=True, verbosity=-1)
                model.fit(x_fit, transformed)
                values[:, j] = np.sinh(model.predict(x_valid)) * scale
                del model
            save_cache(path, valid_idx, ids, values, names)
            print(f"search {target} fold {fold + 1}/5: "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
            del fit_rows, valid_rows, x_fit, x_valid, y, transformed
            gc.collect()
        else:
            print(f"search {target} fold {fold + 1}/5: loaded", flush=True)
        output[valid_idx] = values
    save_complete(OUTPUT / f"{target}_oof.parquet", data, output, names)
    rows = []
    for j, name in enumerate(names):
        folds_ = [score_one(data.loc[assignment == fold], target,
                            output[assignment == fold, j]) for fold in range(5)]
        rows.append({"target": target, "config": name,
                     "smape_percent": score_one(data, target, output[:, j]),
                     **{f"fold_{k + 1}_smape_percent": value for k, value in enumerate(folds_)}})
    results = pd.DataFrame(rows).sort_values("smape_percent")
    results.to_csv(OUTPUT / f"{target}_scores.csv", index=False)
    baseline = results.set_index("config").loc["baseline"]
    best = results.iloc[0]
    wins = sum(best[f"fold_{k}_smape_percent"] < baseline[f"fold_{k}_smape_percent"]
               for k in range(1, 6))
    stable = bool(best["config"] != "baseline" and wins >= 3 and
                  baseline["smape_percent"] - best["smape_percent"] >= 0.15)
    selected = str(best["config"]) if stable else "baseline"
    return {"target": target, "selected_config": selected,
            "development_smape_percent": float(results.set_index("config").loc[selected, "smape_percent"]),
            "baseline_smape_percent": float(baseline["smape_percent"]),
            "fold_wins_vs_baseline": int(wins) if stable else 0,
            "feature_set": "base + target lags" if engineered is not None else "base"}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    data = load_train()
    split = folds(data)
    selected = eligible_targets()
    print("Arcsinh-beneficial targets:", selected, flush=True)
    decisions = [run_target(target, data, split) for target in selected]
    pd.DataFrame(decisions).to_csv(OUTPUT / "development_selection.csv", index=False)
    (OUTPUT / "search_plan.json").write_text(json.dumps({
        "development_seed": 2026, "reserved_validation_seed": 314159,
        "eligibility": "arcsinh beats raw by >=0.15 pp and on >=3/5 folds",
        "selection": "best config only if >=0.15 pp and >=3/5 fold wins versus baseline",
        "configs": CONFIGS, "eligible_targets": selected}, indent=2) + "\n")


if __name__ == "__main__":
    main()
