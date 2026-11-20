import pandas as pd
from feast import FeatureStore
from sklearn.cluster import KMeans

store = FeatureStore(repo_path="feature_repo")
src = pd.read_parquet("data/features/digits_features.parquet")
service = store.get_feature_service("digits_feature_service")

df = store.get_historical_features(
    entity_df=src[["sample_id", "event_timestamp"]], features=service
).to_df()

X = df.drop(columns=["sample_id", "event_timestamp", "target"])
labels = KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(X)

print("=" * 60, "\nFEATURE SERVICE REUSE: KMEANS CLUSTERING\n", "=" * 60, sep="")
print("Cluster sizes:\n", pd.Series(labels).value_counts().sort_index())
print("\nCluster vs true digit:\n", pd.crosstab(labels, df["target"]))