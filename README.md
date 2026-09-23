# Deep Learning Financial Performance Prediction

Predict nine Q0 financial indicators from company metadata and Q1–Q10 historical indicators.

## Contents

- `train.csv` and `test_individual.csv`: assignment data, stored with Git LFS.
- `data_dictionary.txt`: column descriptions.
- `individual-pred-rubric.pdf` and `report-rubric.pdf`: assignment rubrics.
- `evaluation_framework.py`: reproducible data audit, train/validation split, rubric sMAPE, and naive mean baseline.
- `AUDIT_README.md` and `audit_outputs/`: audit findings and baseline results.

Install [Git LFS](https://git-lfs.com/) before cloning to retrieve the CSV files. From the repository root, run:

```bash
python evaluation_framework.py
```

The evaluation framework reads the test file only for schema and missing-value checks. It does not fit or tune a model on test data.
