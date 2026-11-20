import pandas as pd
from feast import FeatureStore

src = pd.read_parquet("data/features/digits_features.parquet")
entity_df = src[["sample_id", "event_timestamp"]].copy()

store = FeatureStore(repo_path="feature_repo")
pixels = [f"digits_measurements:pixel_{i}_{j}" for i in range(8) for j in range(8)]
features = pixels + [
    "digits_measurements:target",
    "digits_engineered_features:pixel_mean",
    "digits_engineered_features:pixel_max",
    "digits_engineered_features:nonzero_pixels",
]
df = store.get_historical_features(entity_df=entity_df, features=features).to_df()
print("=" * 60, "\nFEAST HISTORICAL FEATURE RETRIEVAL\n", "=" * 60, sep="")
print(df[["sample_id", "event_timestamp", "pixel_0_2", "target",
          "pixel_mean", "nonzero_pixels"]].head())
print("Rows retrieved:", len(df))
print("Null cells:", int(df.isna().sum().sum()))