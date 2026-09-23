"""Analyze aligned OOF models, lag baselines, target blends, and identities.

Only train.csv and saved OOF predictions are read. Never reads test_individual.csv.
Run --prelim for lag/identity checks, then --all after model OOF files exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_framework import ID_COLUMN, ROOT, TARGETS, score_predictions
from hard_target_oof import HARD_TARGETS

OUTPUT = ROOT / "oof_outputs"
LAG_METHODS = ("q1", "mean_q1_q2", "mean_q1_q4", "weighted_q1_q4")
RF_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAG_BLEND_WEIGHTS = (0.25, 0.5, 0.75)
DIRECT_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def read_training_subset() -> pd.DataFrame:
    columns = [ID_COLUMN, *TARGETS]
    columns += [f"Q{k}_{target.removeprefix('Q0_')}" for target in TARGETS for k in range(1, 5)]
    data = pd.read_csv(ROOT / "train.csv", usecols=columns)
    if len(data) != 100_000 or not data[ID_COLUMN].is_unique:
        raise ValueError("Unexpected training row count or duplicate Id")
    return data


def target_score(truth: pd.DataFrame, target: str, prediction: np.ndarray) -> float:
    """Score one candidate through the exact shared nine-target rubric function."""
    surrogate = truth.copy(deep=False)
    surrogate[target] = np.asarray(prediction, dtype=np.float64)
    per_target, _ = score_predictions(truth, surrogate)
    return float(per_target[target])


def aligned_frame(data: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    aligned = prediction.copy()
    aligned.insert(0, ID_COLUMN, data[ID_COLUMN].to_numpy())
    aligned.insert(0, "row_index", np.arange(len(data)))
    return aligned


def read_aligned(path: Path, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    saved = pd.read_parquet(path)
    if not np.array_equal(saved.row_index.to_numpy(), np.arange(len(data))) or not np.array_equal(
        saved[ID_COLUMN].to_numpy(), data[ID_COLUMN].to_numpy()
    ):
        raise ValueError(f"OOF row/Id mismatch: {path}")
    if not np.isfinite(saved[columns].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite OOF prediction: {path}")
    return saved[columns]


def lag_predictions(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs = {name: pd.DataFrame(index=data.index, columns=TARGETS, dtype=np.float64)
               for name in LAG_METHODS}
    weights = np.array([0.6, 0.25, 0.10, 0.05])
    for target in TARGETS:
        suffix = target.removeprefix("Q0_")
        lag = data[[f"Q{k}_{suffix}" for k in range(1, 5)]].to_numpy(dtype=np.float64)
        outputs["q1"][target] = lag[:, 0]
        outputs["mean_q1_q2"][target] = lag[:, :2].mean(axis=1)
        outputs["mean_q1_q4"][target] = lag.mean(axis=1)
        outputs["weighted_q1_q4"][target] = lag @ weights
    return outputs


def identity_checks(data: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    relationships = [
        ("assets = liabilities + equity", "Q0_TOTAL_ASSETS",
         data.Q0_TOTAL_LIABILITIES + data.Q0_TOTAL_STOCKHOLDERS_EQUITY),
        ("revenues = gross_profit + cost_of_revenues", "Q0_REVENUES",
         data.Q0_GROSS_PROFIT + data.Q0_COST_OF_REVENUES),
        ("operating_income = gross_profit - operating_expenses", "Q0_OPERATING_INCOME",
         data.Q0_GROSS_PROFIT - data.Q0_OPERATING_EXPENSES),
    ]
    rows = []
    for identity, target, derived in relationships:
        residual = data[target] - derived
        rows.append({
            "identity": identity,
            "target": target,
            "identity_smape_percent": target_score(truth, target, derived.to_numpy()),
            "median_absolute_residual": float(np.median(np.abs(residual))),
            "p95_absolute_residual": float(np.quantile(np.abs(residual), 0.95)),
            "max_absolute_residual": float(np.max(np.abs(residual))),
        })
    return pd.DataFrame(rows)


def run_prelim(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    OUTPUT.mkdir(exist_ok=True)
    truth = data[list(TARGETS)]
    lag = lag_predictions(data)
    rows = []
    for name, prediction in lag.items():
        per_target, average = score_predictions(truth, prediction)
        row = {"method": name, "mean_smape_percent": average}
        row.update({target: float(per_target[target]) for target in TARGETS})
        rows.append(row)
        aligned_frame(data, prediction).to_parquet(
            OUTPUT / f"lag_{name}.parquet", index=False, compression="zstd"
        )
    pd.DataFrame(rows).sort_values("mean_smape_percent").to_csv(
        OUTPUT / "lag_scores.csv", index=False
    )
    identity_checks(data, truth).to_csv(OUTPUT / "identity_checks.csv", index=False)
    print("Lag baselines:\n", pd.DataFrame(rows)[["method", "mean_smape_percent"]].to_string(index=False),
          flush=True)
    return lag


def load_model_oof(data: pd.DataFrame, truth: pd.DataFrame) -> dict[str, pd.DataFrame]:
    models = {}
    rows = []
    for name in ("random_forest", "lightgbm", "hist_gradient_boosting"):
        prediction = read_aligned(OUTPUT / f"{name}_oof.parquet", data, list(TARGETS))
        per_target, mean = score_predictions(truth, prediction)
        recorded = json.loads((OUTPUT / f"{name}_metrics.json").read_text())
        if abs(mean - recorded["mean_smape_percent"]) > 1e-9:
            raise ValueError(f"Saved score mismatch: {name}")
        models[name] = prediction
        row = {"model": name, "mean_smape_percent": mean}
        row.update({target: float(per_target[target]) for target in TARGETS})
        rows.append(row)
    pd.DataFrame(rows).sort_values("mean_smape_percent").to_csv(
        OUTPUT / "base_model_scores.csv", index=False
    )
    return models


def load_hard_oof(data: pd.DataFrame, truth: pd.DataFrame,
                  models: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    hard = {}
    rows = []
    fold_rows = []
    assignment = pd.read_parquet(OUTPUT / "fold_assignments.parquet")
    if not np.array_equal(assignment.row_index.to_numpy(), np.arange(len(data))) or not np.array_equal(
        assignment[ID_COLUMN].to_numpy(), data[ID_COLUMN].to_numpy()
    ):
        raise ValueError("Fold assignments are misaligned")
    fold_ids = assignment.fold.to_numpy()
    for target in HARD_TARGETS:
        saved = read_aligned(OUTPUT / f"hard_{target}_oof.parquet",
                             data, ["raw_lag", "arcsinh_lag"])
        hard[target] = saved
        actual = truth[target].to_numpy(dtype=np.float64)
        for method, prediction in (
            ("lightgbm_base_raw", models["lightgbm"][target].to_numpy()),
            ("lightgbm_raw_lag", saved.raw_lag.to_numpy()),
            ("lightgbm_arcsinh_lag", saved.arcsinh_lag.to_numpy()),
        ):
            row = {"target": target, "method": method,
                   "smape_percent": target_score(truth, target, prediction)}
            if target in ("Q0_OPERATING_INCOME", "Q0_EBITDA"):
                row["sign_accuracy_percent"] = 100 * float(np.mean(
                    np.sign(actual) == np.sign(prediction)
                ))
            rows.append(row)
            for fold in range(5):
                inside = fold_ids == fold
                fold_truth = truth.loc[inside].reset_index(drop=True)
                fold_rows.append({"target": target, "method": method, "fold": fold + 1,
                                  "smape_percent": target_score(fold_truth, target,
                                                                 prediction[inside])})
    pd.DataFrame(rows).to_csv(OUTPUT / "hard_experiment_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "hard_fold_scores.csv", index=False)
    return hard


def select_direct(data: pd.DataFrame, truth: pd.DataFrame,
                  models: dict[str, pd.DataFrame], lag: dict[str, pd.DataFrame],
                  hard: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.DataFrame(index=data.index, columns=TARGETS, dtype=np.float64)
    selections = []
    all_candidates = []
    for target in TARGETS:
        rf = models["random_forest"][target].to_numpy(dtype=np.float64)
        lgb = models["lightgbm"][target].to_numpy(dtype=np.float64)
        candidates: list[tuple[str, np.ndarray, dict]] = []
        rf_lgb = []
        for rf_weight in RF_WEIGHTS:
            prediction = rf_weight * rf + (1 - rf_weight) * lgb
            label = f"rf_lgb_rf_weight_{rf_weight:.2f}"
            score = target_score(truth, target, prediction)
            rf_lgb.append((score, rf_weight, prediction))
            candidates.append((label, prediction, {"rf_weight": rf_weight,
                                                   "lgb_weight": 1 - rf_weight}))
        best_blend_score, best_rf_weight, best_blend_prediction = min(rf_lgb, key=lambda x: x[0])
        candidates.append(("hist_gradient_boosting",
                           models["hist_gradient_boosting"][target].to_numpy(dtype=np.float64), {}))
        for lag_name in LAG_METHODS:
            lag_value = lag[lag_name][target].to_numpy(dtype=np.float64)
            candidates.append((f"lag_{lag_name}", lag_value,
                               {"lag_method": lag_name, "lag_weight": 1.0}))
            for lag_weight in LAG_BLEND_WEIGHTS:
                prediction = lag_weight * lag_value + (1 - lag_weight) * best_blend_prediction
                candidates.append((f"lag_{lag_name}_w{lag_weight:.2f}_plus_rf_lgb",
                                   prediction, {"lag_method": lag_name, "lag_weight": lag_weight,
                                                "rf_weight_in_other_part": best_rf_weight}))
        if target in hard:
            candidates.append(("lightgbm_raw_lag", hard[target].raw_lag.to_numpy(dtype=np.float64), {}))
            candidates.append(("lightgbm_arcsinh_lag",
                               hard[target].arcsinh_lag.to_numpy(dtype=np.float64), {}))
        evaluated = []
        for label, prediction, parameters in candidates:
            score = target_score(truth, target, prediction)
            all_candidates.append({"target": target, "method": label,
                                   "smape_percent": score, **parameters})
            evaluated.append((score, label, prediction, parameters))
        score, label, prediction, parameters = min(evaluated, key=lambda item: item[0])
        selected[target] = prediction
        selections.append({"target": target, "best_rf_weight": best_rf_weight,
                           "best_rf_lgb_smape_percent": best_blend_score,
                           "best_direct_method": label,
                           "best_direct_smape_percent": score,
                           "best_direct_parameters": json.dumps(parameters, sort_keys=True)})
    pd.DataFrame(all_candidates).sort_values(["target", "smape_percent"]).to_csv(
        OUTPUT / "direct_candidates.csv", index=False
    )
    selection_table = pd.DataFrame(selections)
    selection_table.to_csv(OUTPUT / "direct_selection.csv", index=False)
    aligned_frame(data, selected).to_parquet(
        OUTPUT / "selected_direct_oof.parquet", index=False, compression="zstd"
    )
    return selected, selection_table


def accounting_derivations(direct: pd.DataFrame) -> dict[str, list[tuple[str, np.ndarray]]]:
    d = direct
    return {
        "Q0_TOTAL_ASSETS": [("liabilities_plus_equity",
                             (d.Q0_TOTAL_LIABILITIES + d.Q0_TOTAL_STOCKHOLDERS_EQUITY).to_numpy())],
        "Q0_TOTAL_LIABILITIES": [("assets_minus_equity",
                                  (d.Q0_TOTAL_ASSETS - d.Q0_TOTAL_STOCKHOLDERS_EQUITY).to_numpy())],
        "Q0_TOTAL_STOCKHOLDERS_EQUITY": [("assets_minus_liabilities",
                                          (d.Q0_TOTAL_ASSETS - d.Q0_TOTAL_LIABILITIES).to_numpy())],
        "Q0_REVENUES": [("gross_profit_plus_cost",
                         (d.Q0_GROSS_PROFIT + d.Q0_COST_OF_REVENUES).to_numpy())],
        "Q0_GROSS_PROFIT": [
            ("revenues_minus_cost", (d.Q0_REVENUES - d.Q0_COST_OF_REVENUES).to_numpy()),
            ("operating_income_plus_expenses",
             (d.Q0_OPERATING_INCOME + d.Q0_OPERATING_EXPENSES).to_numpy()),
        ],
        "Q0_COST_OF_REVENUES": [("revenues_minus_gross_profit",
                                 (d.Q0_REVENUES - d.Q0_GROSS_PROFIT).to_numpy())],
        "Q0_OPERATING_INCOME": [("gross_profit_minus_expenses",
                                 (d.Q0_GROSS_PROFIT - d.Q0_OPERATING_EXPENSES).to_numpy())],
        "Q0_OPERATING_EXPENSES": [("gross_profit_minus_operating_income",
                                   (d.Q0_GROSS_PROFIT - d.Q0_OPERATING_INCOME).to_numpy())],
    }


def accounting_model_experiment(truth: pd.DataFrame,
                                models: dict[str, pd.DataFrame]) -> None:
    """Direct/derived/grid blends using each base model's OOF values alone."""
    rows = []
    for model_name, prediction in models.items():
        for target, formulas in accounting_derivations(prediction).items():
            direct = prediction[target].to_numpy(dtype=np.float64)
            for formula, derived in formulas:
                for weight in DIRECT_WEIGHTS:
                    candidate = weight * direct + (1 - weight) * derived
                    rows.append({"model": model_name, "target": target,
                                 "derived_formula": formula,
                                 "direct_weight": weight,
                                 "smape_percent": target_score(truth, target, candidate)})
    pd.DataFrame(rows).to_csv(OUTPUT / "accounting_model_candidates.csv", index=False)


def select_accounting(data: pd.DataFrame, truth: pd.DataFrame,
                      direct: pd.DataFrame, direct_selection: pd.DataFrame) -> pd.DataFrame:
    derived = accounting_derivations(direct)
    final = direct.copy()
    candidates = []
    recipes = []
    direct_lookup = direct_selection.set_index("target")
    for target in TARGETS:
        direct_value = direct[target].to_numpy(dtype=np.float64)
        direct_score = target_score(truth, target, direct_value)
        best = (direct_score, "direct", 1.0, direct_value)
        candidates.append({"target": target, "derived_formula": "direct",
                           "direct_weight": 1.0, "smape_percent": direct_score})
        for formula, derived_value in derived.get(target, []):
            for weight in DIRECT_WEIGHTS:
                prediction = weight * direct_value + (1 - weight) * derived_value
                score = target_score(truth, target, prediction)
                candidates.append({"target": target, "derived_formula": formula,
                                   "direct_weight": weight, "smape_percent": score})
                if score < best[0]:
                    best = (score, formula, weight, prediction)
        score, formula, weight, prediction = best
        nonnegative_guard = target in ("Q0_REVENUES", "Q0_OPERATING_EXPENSES")
        clipped_count = int((prediction < 0).sum()) if nonnegative_guard else 0
        if nonnegative_guard:
            # These two labels are nonnegative in every training fold. For a
            # positive actual, a negative or zero prediction both score 200%.
            prediction = np.maximum(prediction, 0.0)
            score = target_score(truth, target, prediction)
        final[target] = prediction
        recipes.append({
            "target": target,
            "direct_method": direct_lookup.loc[target, "best_direct_method"],
            "direct_parameters": direct_lookup.loc[target, "best_direct_parameters"],
            "direct_smape_percent": direct_score,
            "derived_formula": formula,
            "direct_weight": weight,
            "nonnegative_guard": nonnegative_guard,
            "clipped_negative_count": clipped_count,
            "final_smape_percent": score,
        })
    pd.DataFrame(candidates).sort_values(["target", "smape_percent"]).to_csv(
        OUTPUT / "accounting_candidates.csv", index=False
    )
    recipe = pd.DataFrame(recipes)
    recipe.to_csv(OUTPUT / "target_recipes.csv", index=False)
    aligned_frame(data, final).to_parquet(
        OUTPUT / "selected_final_oof.parquet", index=False, compression="zstd"
    )
    direct_per_target, direct_mean = score_predictions(truth, direct)
    final_per_target, final_mean = score_predictions(truth, final)
    summary = {"direct_selected_mean_smape_percent": direct_mean,
               "final_selected_mean_smape_percent": final_mean,
               "direct_target_smape_percent": direct_per_target.to_dict(),
               "final_target_smape_percent": final_per_target.to_dict(),
               "selection_note": "Recipes/weights selected on same OOF labels; post-selection scores optimistic."}
    (OUTPUT / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assignment = pd.read_parquet(OUTPUT / "fold_assignments.parquet")
    fold_rows = []
    for fold in range(5):
        inside = assignment.fold.to_numpy() == fold
        per_target, mean = score_predictions(truth.loc[inside], final.loc[inside])
        fold_row = {"fold": fold + 1, "mean_smape_percent": mean}
        fold_row.update({target: float(per_target[target]) for target in TARGETS})
        fold_rows.append(fold_row)
    pd.DataFrame(fold_rows).to_csv(OUTPUT / "selected_final_fold_scores.csv", index=False)
    print(f"Selected direct OOF mean: {direct_mean:.5f}%", flush=True)
    print(f"Selected after accounting blends: {final_mean:.5f}%", flush=True)
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prelim", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    data = read_training_subset()
    lag = run_prelim(data)
    if args.all:
        truth = data[list(TARGETS)]
        models = load_model_oof(data, truth)
        accounting_model_experiment(truth, models)
        hard = load_hard_oof(data, truth, models)
        direct, selection = select_direct(data, truth, models, lag, hard)
        recipe = select_accounting(data, truth, direct, selection)
        print(recipe[["target", "direct_method", "derived_formula", "direct_weight",
                      "final_smape_percent"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
