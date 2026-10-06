"""Descriptive uncertainty from saved predictions only; no fitting or selection."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "Q0_TOTAL_ASSETS", "Q0_TOTAL_LIABILITIES", "Q0_TOTAL_STOCKHOLDERS_EQUITY",
    "Q0_GROSS_PROFIT", "Q0_COST_OF_REVENUES", "Q0_REVENUES",
    "Q0_OPERATING_INCOME", "Q0_OPERATING_EXPENSES", "Q0_EBITDA",
)
MODELS = ("mean", "lightgbm", "neural_network")
BOOTSTRAPS = 2000
SEED = 20261006


def main():
    truth = pd.read_csv(ROOT / "benchmark_outputs/validation_truth.csv")
    actual = truth[list(TARGETS)].to_numpy(dtype=np.float64)
    errors = []
    for model in MODELS:
        saved = pd.read_csv(ROOT / f"benchmark_outputs/validation_predictions/{model}.csv")
        if not saved[["row_index", "Id"]].equals(truth[["row_index", "Id"]]):
            raise ValueError("Saved validation prediction alignment differs")
        prediction = saved[list(TARGETS)].to_numpy(dtype=np.float64)
        denominator = 0.5 * (np.abs(actual) + np.abs(prediction))
        terms = np.divide(np.abs(actual - prediction), denominator,
                          out=np.zeros_like(denominator), where=denominator != 0)
        errors.append(100.0 * terms.mean(axis=1))
    row_errors = np.column_stack(errors)
    rng = np.random.default_rng(SEED)
    boot = np.empty((BOOTSTRAPS, len(MODELS)))
    for start in range(0, BOOTSTRAPS, 50):
        stop = min(start + 50, BOOTSTRAPS)
        indices = rng.integers(0, len(truth), size=(stop - start, len(truth)))
        boot[start:stop] = row_errors[indices].mean(axis=1)
    result = {"seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
              "validation_rows": len(truth), "models": {}}
    for j, model in enumerate(MODELS):
        ci = np.quantile(boot[:, j], [0.025, 0.975])
        result["models"][model] = {
            "mean_smape_percent": float(row_errors[:, j].mean()),
            "conditional_bootstrap_95_interval_percent": ci.tolist(),
        }
    difference = boot[:, 2] - boot[:, 1]
    result["neural_minus_lightgbm"] = {
        "difference_percentage_points": float((row_errors[:, 2] - row_errors[:, 1]).mean()),
        "paired_bootstrap_95_interval_percentage_points": np.quantile(
            difference, [0.025, 0.975]).tolist(),
    }
    folds = pd.read_csv(ROOT / "candidate_cv_314159/held_fold_scores.csv")
    for name in ("locked", "candidate"):
        values = folds[f"{name}_mean_smape_percent"].to_numpy()
        result[f"{name}_seed314159_fold_summary"] = {
            "mean_smape_percent": float(values.mean()),
            "sample_sd_percentage_points": float(values.std(ddof=1)),
            "range_percent": [float(values.min()), float(values.max())],
        }
    result["limitations"] = (
        "Bootstrap intervals condition on fixed fitted models and exchangeable validation rows; "
        "they do not include model selection/training variability. Different CV seeds reuse "
        "the same rows and are not independent holdout samples."
    )
    (ROOT / "individual_report_outputs/uncertainty_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
