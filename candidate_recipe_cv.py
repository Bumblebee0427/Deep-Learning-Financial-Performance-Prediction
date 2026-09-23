"""Select a small model substitution on development OOF, then validate on seed 314159.

The original accounting weights stay fixed. Selection and validation outputs
are stored separately; neither stage reads test_individual.csv.
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from benchmark_models import TREE_THREADS, matrices
from evaluation_framework import ID_COLUMN, ROOT, TARGETS, predictor_columns, score_predictions
from hard_target_oof import HARD_TARGETS, lag_features
from hard_representations_cv import estimator as representation_estimator, transformed_fit_predict
from lightgbm_small_search import CONFIGS
from locked_recipe_cv import apply_accounting_recipe, checked_cache, load_train, read_aligned, save_cache, save_complete

DEV_LOCKED = ROOT / "locked_cv_2026"
HELD_LOCKED = ROOT / "locked_cv_314159"
DEV_REP = ROOT / "representation_cv_2026"
DEV_ARC = ROOT / "ablation_cv_2026"
DEV_SEARCH = ROOT / "search_cv_2026"
OUTPUT = ROOT / "candidate_cv_314159"
CACHE = OUTPUT / "fold_cache"
FROZEN = OUTPUT / "selected_substitutions.json"


def recipe() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "oof_outputs" / "target_recipes.csv")
    return frame.set_index("target").loc[list(TARGETS)]


def method_predictions(data: pd.DataFrame, target: str, method: str,
                       *, held: bool) -> np.ndarray:
    if method == "locked":
        root = HELD_LOCKED if held else DEV_LOCKED
        return read_aligned(root / "locked_direct_oof.parquet", data, list(TARGETS))[target].to_numpy()
    if held:
        return read_aligned(OUTPUT / f"{target}_{method}_oof.parquet", data,
                            [target])[target].to_numpy()
    if method == "arc_base":
        return read_aligned(DEV_ARC / "arcsinh_oof.parquet", data, list(TARGETS))[target].to_numpy()
    if method == "sign_magnitude":
        return read_aligned(DEV_REP / f"{target}_oof.parquet", data,
                            ["sign_magnitude"])["sign_magnitude"].to_numpy()
    if method.startswith("search_"):
        config = method.removeprefix("search_")
        return read_aligned(DEV_SEARCH / f"{target}_oof.parquet", data,
                            [config])[config].to_numpy()
    raise ValueError(method)


def fold_mean_scores(data: pd.DataFrame, final: pd.DataFrame,
                     assignments: np.ndarray) -> list[float]:
    truth = data[list(TARGETS)]
    return [score_predictions(truth.loc[assignments == fold],
                              final.loc[assignments == fold])[1] for fold in range(5)]


def select() -> None:
    OUTPUT.mkdir(exist_ok=True)
    data = load_train()
    weights = recipe()
    direct = read_aligned(DEV_LOCKED / "locked_direct_oof.parquet", data, list(TARGETS)).copy()
    assignments = pd.read_parquet(DEV_LOCKED / "fold_assignments.parquet").fold.to_numpy()
    search = pd.read_csv(DEV_SEARCH / "development_selection.csv").set_index("target")
    decisions = []
    for target in TARGETS:
        choices = ["arc_base"]
        if target in search.index and search.loc[target, "selected_config"] != "baseline":
            choices.append("search_" + str(search.loc[target, "selected_config"]))
        if target in ("Q0_OPERATING_INCOME", "Q0_EBITDA"):
            choices.append("sign_magnitude")
        old_final = apply_accounting_recipe(direct, weights)
        old_score = score_predictions(data[list(TARGETS)], old_final)[1]
        old_folds = fold_mean_scores(data, old_final, assignments)
        chosen = "locked"
        chosen_score = old_score
        chosen_folds = old_folds
        for method in choices:
            trial = direct.copy()
            trial[target] = method_predictions(data, target, method, held=False)
            final = apply_accounting_recipe(trial, weights)
            score = score_predictions(data[list(TARGETS)], final)[1]
            folds_ = fold_mean_scores(data, final, assignments)
            wins = sum(a < b for a, b in zip(folds_, old_folds))
            if old_score - score >= 0.05 and wins >= 3 and score < chosen_score:
                chosen, chosen_score, chosen_folds = method, score, folds_
        if chosen != "locked":
            direct[target] = method_predictions(data, target, chosen, held=False)
        decisions.append({"target": target, "method": chosen,
                          "dev_mean_smape_before": old_score,
                          "dev_mean_smape_after": chosen_score,
                          "fold_wins": sum(a < b for a, b in zip(chosen_folds, old_folds))})
        print(f"candidate {target}: {chosen} ({old_score:.5f} -> {chosen_score:.5f})", flush=True)
    final = apply_accounting_recipe(direct, weights)
    save_complete(OUTPUT / "development_candidate_oof.parquet", data,
                  final.to_numpy(dtype=np.float64), list(TARGETS))
    baseline = read_aligned(DEV_LOCKED / "locked_final_oof.parquet", data, list(TARGETS))
    old_targets, _ = score_predictions(data[list(TARGETS)], baseline)
    new_targets, _ = score_predictions(data[list(TARGETS)], final)
    pd.DataFrame({"target": TARGETS,
                  "locked_smape_percent": old_targets.to_numpy(),
                  "candidate_smape_percent": new_targets.to_numpy(),
                  "improvement_pp": old_targets.to_numpy() - new_targets.to_numpy(),
                  }).to_csv(OUTPUT / "development_target_scores.csv", index=False)
    old_folds = fold_mean_scores(data, baseline, assignments)
    new_folds = fold_mean_scores(data, final, assignments)
    pd.DataFrame({"fold": range(1, 6), "locked_mean_smape_percent": old_folds,
                  "candidate_mean_smape_percent": new_folds,
                  "improvement_pp": np.array(old_folds) - np.array(new_folds),
                  }).to_csv(OUTPUT / "development_fold_scores.csv", index=False)
    pd.DataFrame(decisions).to_csv(OUTPUT / "development_decisions.csv", index=False)
    selected = {row["target"]: row["method"] for row in decisions if row["method"] != "locked"}
    FROZEN.write_text(json.dumps({"development_seed": 2026, "held_seed": 314159,
                                  "accounting_recipe": "original frozen seed42 weights",
                                  "selection_rule": "full mean gain >=0.05 pp and >=3/5 fold wins, one pass in rubric target order",
                                  "substitutions": selected}, indent=2) + "\n")
    print(f"development full recipe: {score_predictions(data[list(TARGETS)], final)[1]:.5f}%", flush=True)


def arcsinh_model(fold: int, method: str) -> LGBMRegressor:
    config = (CONFIGS[method.removeprefix("search_")] if method.startswith("search_")
              else CONFIGS["baseline"])
    return LGBMRegressor(**config, colsample_bytree=0.8, n_jobs=TREE_THREADS,
                         random_state=42 + fold, deterministic=True,
                         force_col_wise=True, verbosity=-1)


def fit_held() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    data = load_train()
    choices = json.loads(FROZEN.read_text())["substitutions"]
    split = list(KFold(n_splits=5, shuffle=True, random_state=314159).split(data))
    held_assignments = pd.read_parquet(HELD_LOCKED / "fold_assignments.parquet")
    assignment = np.full(len(data), -1, dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(split):
        assignment[valid_idx] = fold
    if not np.array_equal(assignment, held_assignments.fold.to_numpy()):
        raise ValueError("Held split does not match baseline")
    for target, method in choices.items():
        engineered = lag_features(data, target) if (method == "sign_magnitude" or
                    (method.startswith("search_") and target in HARD_TARGETS)) else None
        full = pd.concat([data, engineered], axis=1) if engineered is not None else data
        features = predictor_columns(data) + (list(engineered.columns) if engineered is not None else [])
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(full[c])]
        categorical = [c for c in features if c not in numeric]
        output = np.full(len(data), np.nan, dtype=np.float64)
        for fold, (fit_idx, valid_idx) in enumerate(split):
            path = CACHE / f"{target}_{method}_fold{fold + 1}.parquet"
            ids = data.iloc[valid_idx][ID_COLUMN].to_numpy()
            saved = checked_cache(path, valid_idx, ids, [target])
            if saved is None:
                started = time.perf_counter()
                fit_rows, valid_rows = full.iloc[fit_idx], full.iloc[valid_idx]
                x_fit, x_valid = matrices(fit_rows, valid_rows, features, numeric,
                                          categorical, scaled=False)
                y = fit_rows[target].to_numpy(dtype=np.float64)
                scale = float(np.median(np.abs(y))) or 1.0
                if method == "sign_magnitude":
                    classifier = representation_estimator(fold, classifier=True)
                    classifier.fit(x_fit, (y > 0).astype(np.int8))
                    positive = classifier.predict_proba(x_valid)[:, 1] >= 0.5
                    magnitude = transformed_fit_predict(x_fit, np.abs(y), x_valid, fold)
                    values = np.where(positive, magnitude, -magnitude)
                else:
                    model = arcsinh_model(fold, method)
                    model.fit(x_fit, np.arcsinh(y / scale))
                    values = np.sinh(model.predict(x_valid)) * scale
                save_cache(path, valid_idx, ids, values.reshape(-1, 1), [target])
                print(f"held {target} {method} fold {fold + 1}/5: "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
                if method != "sign_magnitude":
                    del model
                del fit_rows, valid_rows, x_fit, x_valid
                gc.collect()
            else:
                values = saved[:, 0]
                print(f"held {target} {method} fold {fold + 1}/5: loaded", flush=True)
            output[valid_idx] = values
        save_complete(OUTPUT / f"{target}_{method}_oof.parquet", data,
                      output.reshape(-1, 1), [target])


def validate() -> None:
    data = load_train()
    choices = json.loads(FROZEN.read_text())["substitutions"]
    direct = read_aligned(HELD_LOCKED / "locked_direct_oof.parquet", data, list(TARGETS)).copy()
    baseline = read_aligned(HELD_LOCKED / "locked_final_oof.parquet", data, list(TARGETS))
    for target, method in choices.items():
        direct[target] = method_predictions(data, target, method, held=True)
    candidate = apply_accounting_recipe(direct, recipe())
    save_complete(OUTPUT / "held_candidate_oof.parquet", data,
                  candidate.to_numpy(dtype=np.float64), list(TARGETS))
    truth = data[list(TARGETS)]
    baseline_scores, baseline_mean = score_predictions(truth, baseline)
    candidate_scores, candidate_mean = score_predictions(truth, candidate)
    pd.DataFrame({"target": TARGETS,
                  "locked_smape_percent": baseline_scores.to_numpy(),
                  "candidate_smape_percent": candidate_scores.to_numpy(),
                  "improvement_pp": baseline_scores.to_numpy() - candidate_scores.to_numpy(),
                  }).to_csv(OUTPUT / "held_target_scores.csv", index=False)
    assignments = pd.read_parquet(HELD_LOCKED / "fold_assignments.parquet").fold.to_numpy()
    rows = []
    for fold in range(5):
        mask = assignments == fold
        old = score_predictions(truth.loc[mask], baseline.loc[mask])[1]
        new = score_predictions(truth.loc[mask], candidate.loc[mask])[1]
        rows.append({"fold": fold + 1, "locked_mean_smape_percent": old,
                     "candidate_mean_smape_percent": new, "improvement_pp": old - new})
    pd.DataFrame(rows).to_csv(OUTPUT / "held_fold_scores.csv", index=False)
    (OUTPUT / "held_summary.json").write_text(json.dumps({
        "seed": 314159, "locked_mean_smape_percent": baseline_mean,
        "candidate_mean_smape_percent": candidate_mean,
        "improvement_pp": baseline_mean - candidate_mean,
        "substitutions": choices}, indent=2) + "\n")
    print(f"held baseline={baseline_mean:.5f}%, candidate={candidate_mean:.5f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "fit-held", "validate"))
    stage = parser.parse_args().stage
    if stage == "select":
        select()
    elif stage == "fit-held":
        fit_held()
    else:
        validate()


if __name__ == "__main__":
    main()
