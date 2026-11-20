# MLOps-Practical-Exam
MLOps practical examination repository

## Aim: Handwritten Digit Classification
End-to-end MLOps pipeline on the Scikit-learn Digits dataset (1797 samples, 64 pixel features, 10 classes).

- Data versioning: DVC (`data/raw/dataset.csv`, local remote)
- Data pipeline: `src/preprocess.py` + `dvc.yaml`
- Feature store: Feast (`feature_repo/`)
- Experiment tracking: MLflow (`src/train.py`)

Regenerate the raw data with: `python src/load_data.py`
