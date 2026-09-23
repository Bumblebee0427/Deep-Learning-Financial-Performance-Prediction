"""Re-evaluate the frozen seed-42 recipe on a new seed-2026 five-fold split.

Only train.csv is read. The recipe hash is fixed below and no choices are
re-optimized. Run components separately for memory control, then --assemble.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from analyze_oof import accounting_derivations
from benchmark_models import matrices
from evaluation_framework import ID_COLUMN, ROOT, TARGETS, predictor_columns, score_predictions
from hard_target_oof import HARD_TARGETS, fit_lightgbm, lag_features
from oof_models import train_predict_fold

SEED = int(os.environ.get("LOCKED_CV_SEED", "2026"))
N_SPLITS = 5
OUTPUT = ROOT / os.environ.get("LOCKED_CV_OUTPUT", f"locked_cv_{SEED}")
CACHE = OUTPUT / "fold_cache"
SOURCE_RECIPE = ROOT / "oof_outputs" / "target_recipes.csv"
FROZEN_RECIPE = OUTPUT / "target_recipes_frozen.csv"
RECIPE_SHA256 = "c7546ae4a87d6b561156cd0d050085e3485a12865db604ac1e0a2385640d8a34"


def freeze_recipe() -> pd.DataFrame:
    OUTPUT.mkdir(exist_ok=True)
    if not FROZEN_RECIPE.exists():
        if hashlib.sha256(SOURCE_RECIPE.read_bytes()).hexdigest() != RECIPE_SHA256:
            raise ValueError("Original recipe changed before freezing")
        shutil.copyfile(SOURCE_RECIPE, FROZEN_RECIPE)
    if hashlib.sha256(FROZEN_RECIPE.read_bytes()).hexdigest() != RECIPE_SHA256:
        raise ValueError("Frozen recipe hash mismatch")
    recipe = pd.read_csv(FROZEN_RECIPE)
    if set(recipe.target) != set(TARGETS) or len(recipe) != len(TARGETS):
        raise ValueError("Frozen recipe target set is invalid")
    return recipe.set_index("target").loc[list(TARGETS)]


def load_train() -> pd.DataFrame:
    data = pd.read_csv(ROOT / "train.csv", low_memory=False)
    if len(data) != 100_000 or not data[ID_COLUMN].is_unique:
        raise ValueError("Unexpected training data")
    return data


def folds(data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    split = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(data))
    assignment = np.full(len(data), -1, dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(split):
        assignment[valid_idx] = fold
    if (assignment < 0).any():
        raise AssertionError("Incomplete folds")
    frame = pd.DataFrame({"row_index": np.arange(len(data)), ID_COLUMN: data[ID_COLUMN],
                          "fold": assignment})
    path = OUTPUT / "fold_assignments.parquet"
    if path.exists():
        previous = pd.read_parquet(path)
        if not previous.equals(frame):
            raise ValueError("Existing fold assignment mismatch")
    else:
        frame.to_parquet(path, index=False, compression="zstd")
    return split


def checked_cache(path: Path, valid_idx: np.ndarray, ids: np.ndarray,
                  prediction_columns: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    cached = pd.read_parquet(path)
    if not np.array_equal(cached.row_index.to_numpy(), valid_idx) or not np.array_equal(
        cached[ID_COLUMN].to_numpy(), ids
    ):
        raise ValueError(f"Cache alignment mismatch: {path}")
    return cached[prediction_columns].to_numpy(dtype=np.float64)


def save_cache(path: Path, valid_idx: np.ndarray, ids: np.ndarray,
               values: np.ndarray, columns: list[str]) -> None:
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, ID_COLUMN, ids)
    frame.insert(0, "row_index", valid_idx)
    frame.to_parquet(path, index=False, compression="zstd")


def save_complete(path: Path, data: pd.DataFrame, values: np.ndarray,
                  columns: list[str]) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"Incomplete/non-finite predictions: {path}")
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, ID_COLUMN, data[ID_COLUMN].to_numpy())
    frame.insert(0, "row_index", np.arange(len(data)))
    frame.to_parquet(path, index=False, compression="zstd")


def fit_base(component: str, data: pd.DataFrame,
             split: list[tuple[np.ndarray, np.ndarray]]) -> None:
    name = {"rf": "random_forest", "lgb": "lightgbm"}[component]
    features = predictor_columns(data)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(data[c])]
    categorical = [c for c in features if c not in numeric]
    output = np.full((len(data), len(TARGETS)), np.nan)
    times = []
    for fold, (fit_idx, valid_idx) in enumerate(split):
        path = CACHE / f"{component}_fold{fold + 1}.parquet"
        ids = data.iloc[valid_idx][ID_COLUMN].to_numpy()
        values = checked_cache(path, valid_idx, ids, list(TARGETS))
        if values is None:
            started = time.perf_counter()
            fit_rows, valid_rows = data.iloc[fit_idx], data.iloc[valid_idx]
            x_train, x_valid = matrices(fit_rows, valid_rows, features, numeric, categorical,
                                        scaled=False)
            y_train = fit_rows[list(TARGETS)].to_numpy(dtype=np.float64)
            values = train_predict_fold(name, x_train, y_train, x_valid, fold)
            seconds = time.perf_counter() - started
            save_cache(path, valid_idx, ids, values, list(TARGETS))
            del fit_rows, valid_rows, x_train, x_valid, y_train
            gc.collect()
            print(f"locked {component} fold {fold + 1}/5: {seconds:.1f}s", flush=True)
        else:
            seconds = 0.0
            print(f"locked {component} fold {fold + 1}/5: loaded", flush=True)
        output[valid_idx] = values
        times.append(seconds)
    save_complete(OUTPUT / f"{component}_oof.parquet", data, output, list(TARGETS))
    score, average = score_predictions(data[list(TARGETS)], pd.DataFrame(output, columns=TARGETS))
    (OUTPUT / f"{component}_metrics.json").write_text(json.dumps({
        "seed": SEED, "model": name, "mean_smape_percent": average,
        "target_smape_percent": score.to_dict(), "new_fold_seconds": times,
    }, indent=2) + "\n")
    print(f"locked {component} OOF={average:.5f}%", flush=True)


def fit_hard(data: pd.DataFrame,
             split: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for target in HARD_TARGETS:
        engineered = lag_features(data, target)
        augmented = pd.concat([data, engineered], axis=1)
        features = predictor_columns(data) + list(engineered.columns)
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(augmented[c])]
        categorical = [c for c in features if c not in numeric]
        output = np.full(len(data), np.nan)
        for fold, (fit_idx, valid_idx) in enumerate(split):
            path = CACHE / f"hard_{target}_fold{fold + 1}.parquet"
            ids = data.iloc[valid_idx][ID_COLUMN].to_numpy()
            saved = checked_cache(path, valid_idx, ids, [target])
            if saved is None:
                started = time.perf_counter()
                fit_rows, valid_rows = augmented.iloc[fit_idx], augmented.iloc[valid_idx]
                x_train, x_valid = matrices(fit_rows, valid_rows, features, numeric,
                                            categorical, scaled=False)
                y_train = fit_rows[target].to_numpy(dtype=np.float64)
                values = fit_lightgbm(x_train, y_train, x_valid, fold=fold, transform=True)
                save_cache(path, valid_idx, ids, values.reshape(-1, 1), [target])
                print(f"locked {target} fold {fold + 1}/5: "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
                del fit_rows, valid_rows, x_train, x_valid, y_train
                gc.collect()
            else:
                values = saved[:, 0]
                print(f"locked {target} fold {fold + 1}/5: loaded", flush=True)
            output[valid_idx] = values
        save_complete(OUTPUT / f"hard_{target}_oof.parquet", data,
                      output.reshape(-1, 1), [target])
        del augmented, engineered
        gc.collect()


def read_aligned(path: Path, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not np.array_equal(frame.row_index.to_numpy(), np.arange(len(data))) or not np.array_equal(
        frame[ID_COLUMN].to_numpy(), data[ID_COLUMN].to_numpy()
    ):
        raise ValueError(f"OOF alignment mismatch: {path}")
    return frame[columns]


def apply_accounting_recipe(direct: pd.DataFrame, recipe: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen accounting weights to aligned direct OOF predictions."""
    derived = accounting_derivations(direct)
    final = direct.copy()
    for target, row in recipe.iterrows():
        formula = row.derived_formula
        if formula != "direct":
            sources = dict(derived[target])
            weight = float(row.direct_weight)
            final[target] = weight * direct[target] + (1 - weight) * sources[formula]
        if str(row.nonnegative_guard).lower() == "true":
            final[target] = np.maximum(final[target], 0.0)
    return final


def assemble(data: pd.DataFrame, recipe: pd.DataFrame) -> None:
    rf = read_aligned(OUTPUT / "rf_oof.parquet", data, list(TARGETS))
    lgb = read_aligned(OUTPUT / "lgb_oof.parquet", data, list(TARGETS))
    hard = {target: read_aligned(OUTPUT / f"hard_{target}_oof.parquet", data, [target])[target]
            for target in HARD_TARGETS}
    direct = pd.DataFrame(index=data.index, columns=TARGETS, dtype=np.float64)
    for target, row in recipe.iterrows():
        method = row.direct_method
        params = json.loads(row.direct_parameters)
        if method.startswith("rf_lgb_rf_weight_"):
            weight = float(params["rf_weight"])
            values = weight * rf[target] + (1 - weight) * lgb[target]
        elif method == "lag_q1":
            values = data[f"Q1_{target.removeprefix('Q0_')}"]
        elif method.startswith("lag_q1_w"):
            lag_weight = float(params["lag_weight"])
            rf_weight = float(params["rf_weight_in_other_part"])
            base = rf_weight * rf[target] + (1 - rf_weight) * lgb[target]
            lag = data[f"Q1_{target.removeprefix('Q0_')}"]
            values = lag_weight * lag + (1 - lag_weight) * base
        elif method == "lightgbm_arcsinh_lag":
            values = hard[target]
        else:
            raise ValueError(f"Unrecognized frozen direct method: {method}")
        direct[target] = values

    final = apply_accounting_recipe(direct, recipe)

    save_complete(OUTPUT / "locked_direct_oof.parquet", data,
                  direct.to_numpy(dtype=np.float64), list(TARGETS))
    save_complete(OUTPUT / "locked_final_oof.parquet", data,
                  final.to_numpy(dtype=np.float64), list(TARGETS))
    truth = data[list(TARGETS)]
    per_target, average = score_predictions(truth, final)
    pd.DataFrame({"target": TARGETS,
                  "seed42_selected_smape_percent": recipe.final_smape_percent.to_numpy(),
                  f"seed{SEED}_locked_smape_percent": per_target.to_numpy()}).to_csv(
        OUTPUT / "locked_target_scores.csv", index=False
    )
    assignments = pd.read_parquet(OUTPUT / "fold_assignments.parquet")
    fold_rows = []
    for fold in range(N_SPLITS):
        mask = assignments.fold.to_numpy() == fold
        scores, mean = score_predictions(truth.loc[mask], final.loc[mask])
        fold_rows.append({"fold": fold + 1, "mean_smape_percent": mean,
                          **scores.to_dict()})
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "locked_fold_scores.csv", index=False)
    old_average = float(recipe.final_smape_percent.mean())
    (OUTPUT / "locked_summary.json").write_text(json.dumps({
        "recipe_sha256": RECIPE_SHA256, "cv_seed": SEED,
        "old_seed42_selected_mean_smape_percent": old_average,
        "new_locked_mean_smape_percent": average,
        "difference_percentage_points": average - old_average,
        "target_smape_percent": per_target.to_dict(),
    }, indent=2) + "\n")
    print(f"Frozen recipe: seed42={old_average:.5f}%, seed{SEED}={average:.5f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("rf", "lgb", "hard"))
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if (args.component is None) == (not args.assemble):
        parser.error("choose exactly one of --component or --assemble")
    recipe = freeze_recipe()
    data = load_train()
    split = folds(data)
    CACHE.mkdir(exist_ok=True)
    if args.assemble:
        assemble(data, recipe)
    elif args.component == "hard":
        fit_hard(data, split)
    else:
        fit_base(args.component, data, split)


if __name__ == "__main__":
    main()
