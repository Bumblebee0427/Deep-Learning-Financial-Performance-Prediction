# Deep Learning Financial Performance Prediction

Predict nine Q0 financial indicators from company metadata and Q1–Q10 historical indicators.

## Contents

- `train.csv` and `test_individual.csv`: assignment data, stored with Git LFS.
- `data_dictionary.txt`: column descriptions.
- `individual-pred-rubric.pdf` and `report-rubric.pdf`: assignment rubrics.
- `evaluation_framework.py`: reproducible data audit, train/validation split, rubric sMAPE, and naive mean baseline.
- `AUDIT_README.md` and `audit_outputs/`: audit findings and baseline results.
- `benchmark_models.py` and `benchmark_outputs/`: seven-model validation benchmark and aligned out-of-sample predictions.
- `oof_models.py`, `hard_target_oof.py`, `analyze_oof.py`, and `oof_outputs/REPORT.md`: five-fold OOF models, lag baselines, accounting identities, and target-specific recipes.
- `locked_recipe_cv.py` and `locked_cv_2026/`: exact seed-42 recipe frozen by SHA-256 and reapplied on new seed-2026 folds.
- `arcsinh_ablation_cv.py`, `hard_representations_cv.py`, and their output folders: common-fold target-transform and hard-target representation experiments.
- `lightgbm_small_search.py`, `candidate_recipe_cv.py`, and their output folders: limited development search and seed-314159 recipe check.
- `final_submission.py`: fit the frozen candidate recipe on all 100,000 training rows and write `final_submission.csv` plus distribution diagnostics. It does not run CV or tune on test data.

Install [Git LFS](https://git-lfs.com/) before cloning to retrieve the CSV files. From the repository root, run:

```bash
python evaluation_framework.py
python benchmark_models.py --all
python oof_models.py --all
python hard_target_oof.py --all
python analyze_oof.py --all
python locked_recipe_cv.py --component rf
python locked_recipe_cv.py --component lgb
python locked_recipe_cv.py --component hard
python locked_recipe_cv.py --assemble
python arcsinh_ablation_cv.py
python hard_representations_cv.py
python lightgbm_small_search.py
python candidate_recipe_cv.py select
LOCKED_CV_SEED=314159 python locked_recipe_cv.py --component rf
LOCKED_CV_SEED=314159 python locked_recipe_cv.py --component lgb
LOCKED_CV_SEED=314159 python locked_recipe_cv.py --component hard
LOCKED_CV_SEED=314159 python locked_recipe_cv.py --assemble
python candidate_recipe_cv.py fit-held
python candidate_recipe_cv.py validate
python final_submission.py
```

The CV experiments read only `train.csv`. The earlier evaluation framework read the test file only for schema and missing-value checks. The final submission script reads `test_individual.csv` only to make predictions after the recipe is fixed.
