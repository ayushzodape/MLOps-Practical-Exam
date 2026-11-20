from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float64, Int64
from feast.value_type import ValueType

sample = Entity(name="sample_id", join_keys=["sample_id"], value_type=ValueType.INT64)

digits_source = FileSource(
    name="digits_features_source",
    path="../data/features/digits_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

pixel_fields = [Field(name=f"pixel_{i}_{j}", dtype=Float64) for i in range(8) for j in range(8)]
pixel_fields.append(Field(name="target", dtype=Int64))

digits_measurements_fv = FeatureView(
    name="digits_measurements",
    entities=[sample],
    ttl=timedelta(days=3650),
    schema=pixel_fields,
    source=digits_source,
    online=True,
)

digits_engineered_fv = FeatureView(
    name="digits_engineered_features",
    entities=[sample],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="pixel_mean", dtype=Float64),
        Field(name="pixel_max", dtype=Float64),
        Field(name="nonzero_pixels", dtype=Int64),
    ],
    source=digits_source,
    online=True,
)

digits_feature_service = FeatureService(
    name="digits_feature_service",
    features=[digits_measurements_fv, digits_engineered_fv],
)