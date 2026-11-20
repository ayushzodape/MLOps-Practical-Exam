# Data Pipeline: Handwritten Digit Classification

## Flow
```
data/raw/dataset.csv (DVC tracked, collected by src/load_data.py)
        |
   [preprocess] -> data/processed/digits_preprocessed.csv
        |
   [features]   -> data/processed/digits_features.csv
        |
   [validate]   -> data/validated/validation_report.json
        |
   [split]      -> data/processed/train.csv, test.csv
```

## Pipeline Stages

### Collection (Member 1)
- Purpose: load the Scikit-learn Digits dataset (1797 samples, 8x8 pixels, 10 classes).
- Script: src/load_data.py. Output: data/raw/dataset.csv, versioned with DVC (dataset.csv.dvc).

### 1. preprocess (src/pipeline/preprocess.py)
- Purpose: clean the raw data.
- Input: data/raw/dataset.csv. Output: data/processed/digits_preprocessed.csv.
- Actions: drop duplicate rows, convert pixels to numeric, fill missing pixels with the median, drop rows with no target.

### 2. features (src/pipeline/features.py)
- Purpose: derive new features from the 64 pixels.
- Input: digits_preprocessed.csv. Output: digits_features.csv (71 columns).
- New features: pixel_sum, nonzero_pixels, top_half_sum, bottom_half_sum, left_half_sum, right_half_sum.

### 3. validate (src/pipeline/validate.py)
- Purpose: block bad data before it reaches training.
- Input: digits_features.csv. Output: data/validated/validation_report.json.
- Rules: all expected columns present; no null values; target in 0-9; every class at least 5% of rows; pixels in 0-16; engineered features within their possible ranges.
- On failure the stage exits with code 1 and the pipeline stops.

### 4. split (src/pipeline/split.py)
- Purpose: stratified train/test split and scaling.
- Input: digits_features.csv. Output: data/processed/train.csv, test.csv.
- Parameters (params.yaml): split.test_size, split.random_state. StandardScaler is fitted on train only.

## DVC
dvc.yaml declares each stage's cmd, deps and outs. dvc repro re-runs only stages whose
dependencies changed (content hashes in dvc.lock). dvc dag shows the graph.