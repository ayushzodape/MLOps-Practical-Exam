import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
features = [
    "digits_measurements:pixel_0_2",
    "digits_measurements:pixel_0_3",
    "digits_measurements:pixel_1_4",
    "digits_measurements:target",
    "digits_engineered_features:pixel_mean",
    "digits_engineered_features:pixel_max",
    "digits_engineered_features:nonzero_pixels",
]
res = store.get_online_features(
    features=features,
    entity_rows=[{"sample_id": 0}, {"sample_id": 1}, {"sample_id": 2}],
).to_dict()
print("=" * 60, "\nFEAST ONLINE FEATURE RETRIEVAL\n", "=" * 60, sep="")
print(pd.DataFrame(res).to_string(index=False))