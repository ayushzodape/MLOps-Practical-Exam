import os
import pandas as pd

TRAIN, TEST = "data/processed/train.csv", "data/processed/test.csv"
RAW = "data/raw/dataset.csv"
OUT = "data/features/digits_features.parquet"
PIXELS = [f"pixel_{i}_{j}" for i in range(8) for j in range(8)]

if os.path.exists(TRAIN) and os.path.exists(TEST):
    df = pd.concat([pd.read_csv(TRAIN), pd.read_csv(TEST)], ignore_index=True)
    print("Using processed data (Member 2)")
else:
    df = pd.read_csv(RAW)
    print("Processed data not found, using raw data")

missing = [c for c in PIXELS + ["target"] if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing[:5]} ...")

df = df[PIXELS + ["target"]].copy()
df[PIXELS] = df[PIXELS].astype("float64")
df["target"] = df["target"].round().astype("int64")

# Engineered features
df["pixel_mean"] = df[PIXELS].mean(axis=1)
df["pixel_max"] = df[PIXELS].max(axis=1)
df["nonzero_pixels"] = (df[PIXELS] != 0).sum(axis=1).astype("int64")

df.insert(0, "sample_id", range(len(df)))
start = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
df["event_timestamp"] = pd.date_range(start=start, periods=len(df), freq="min")
df["created_timestamp"] = df["event_timestamp"]

os.makedirs("data/features", exist_ok=True)
df.to_parquet(OUT, index=False)
print("Saved", df.shape, "->", OUT)