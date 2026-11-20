# Feature Store Analysis: Handwritten Digit Classification

## 1. Purpose

This document explains how a feature store (Feast) is used in the Handwritten Digit Classification pipeline, why it was added, and what it gives us compared with reading CSV files directly.

The pipeline works on the Scikit-learn Digits dataset: 1797 samples of 8x8 grayscale images, giving 64 pixel features (`pixel_0_0` to `pixel_7_7`) and a `target` label from 0 to 9.

## 2. Where the feature store fits in the pipeline

```
load_data.py  ->  data/raw/dataset.csv   (versioned with DVC)
      |
preprocess.py ->  data/processed/train.csv, test.csv   (DVC stage)
      |
build_features.py -> feature parquet (adds sample_id + event_timestamp,
                     computes pixel_mean, pixel_max, nonzero_pixels)
      |
feast apply / materialize -> registry + offline store + online store
      |
get_historical_features.py  -> training data (offline)
get_online_features.py      -> low-latency lookup (online)
reuse_features_for_clustering.py -> same features, second task
      |
train.py (MLflow) -> experiments, model selection
```

## 3. Feast objects defined in this project

Configuration is in `feature_repo/feature_store.yaml` and definitions are in `feature_repo/features.py`.

| Component | Name in this project | Details |
|---|---|---|
| **Project** | `digits_project` | Namespace for all objects |
| **Entity** | `sample_id` | Join key identifying one image sample (INT64) |
| **Data source** | `digits_features_source` | `FileSource` reading the parquet written by `src/build_features.py` |
| **Feature view 1** | `digits_measurements` | The 64 raw pixel features `pixel_i_j` (Float64) plus the `target` label (Int64) |
| **Feature view 2** | `digits_engineered_features` | Derived features: `pixel_mean` (Float64), `pixel_max` (Float64), `nonzero_pixels` (Int64) |
| **Feature service** | `digits_feature_service` | Named bundle of the feature views, requested by models |
| **Registry** | `data/registry.db` | Metadata store created by `feast apply` (path is relative to `feature_repo/`) |
| **Online store** | SQLite, `data/online_store.db` | Latest feature values for fast lookup |
| **Offline store** | Local file-based (provider `local`) | Historical values used to build training sets |

## 4. Why two feature views?

The features are split by how they are produced:

- **`digits_measurements`** holds the raw values as they come from the dataset (pixel intensities), plus the label.
- **`digits_engineered_features`** holds values computed from those pixels: the mean intensity, the maximum intensity and the number of non-zero pixels. These summarise how much ink a digit contains and how dark it is, which gives models a compact signal beyond individual pixels.

Keeping raw and derived features in separate views means each can be changed, reused or recomputed independently, and a model can request only the group it needs.

## 5. Workflow

```bash
# 1. Create the feature parquet from the processed data
python src/build_features.py

# 2. Register entity, feature views and service
cd feature_repo
feast apply

# 3. Load the latest feature values into the online store
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
cd ..

# 4. Retrieve features
python src/get_historical_features.py   # offline: training dataset
python src/get_online_features.py       # online: single-sample lookup
python src/reuse_features_for_clustering.py   # reuse for a second task
```

## 6. Offline vs online retrieval

**Offline (historical) retrieval** takes an entity dataframe (`sample_id` plus timestamps) and returns the matching feature values as of those timestamps. It is used to build training datasets, and it is point-in-time correct: a row only sees feature values that existed at or before its timestamp, which prevents data leakage.

**Online retrieval** returns the most recent materialized value for a given `sample_id` in milliseconds. It is what a deployed model would call at prediction time.

Both paths read from the same definitions, so the model sees features computed the same way in training and in serving. This avoids training-serving skew.

**Note on the label:** `target` is stored in `digits_measurements` so that a training set can be built in one retrieval call. When training, it must be separated from the input features, and an online prediction request should ask only for feature columns and never for `target`.

## 7. Feature reuse

The same registered features are reused for a second task, clustering, in `src/reuse_features_for_clustering.py`. The clustering script asks the feature store for the features it needs instead of repeating the preprocessing. This shows the main benefit of a feature store: define once, use in many models.

## 8. Benefits observed

- **Single source of truth:** features are defined in `features.py` and not copied between scripts.
- **Consistency:** training and inference use identical definitions.
- **Reuse:** classification and clustering share the same features.
- **Point-in-time correctness:** historical retrieval avoids leakage.
- **Versioned definitions:** feature definitions are code tracked in Git, while the underlying data is tracked by DVC.
- **Separation of concerns:** data preparation (DVC pipeline), feature serving (Feast) and experiments (MLflow) are independent stages.

## 9. Limitations and trade-offs

- **Local setup only:** the SQLite online store and file-based offline store suit a lab project, not production. A production setup would use something like Redis or DynamoDB for online serving and a data warehouse for offline storage.
- **Synthetic timestamps:** the Digits dataset has no real event time, so `event_timestamp` is generated. In real data it would come from the event itself.
- **Simple engineered features:** `pixel_mean`, `pixel_max` and `nonzero_pixels` ignore spatial structure. Row and column intensity profiles or edge-based features would capture more about the shape of a digit.
- **Small dataset:** with 1797 rows, the latency and scale benefits of a feature store are not visible. The value here is demonstrating the workflow.
- **Extra moving parts:** it adds a registry, materialization and configuration that must be kept in sync with the data.

## 10. Conclusion

Adding Feast turns the feature columns from files that each script reads in its own way into managed, versioned, reusable assets. Training reads them offline, a serving application would read them online, and both use the same definitions. Together with DVC for data versioning and MLflow for experiment tracking, this completes the data, features and model parts of the MLOps pipeline.