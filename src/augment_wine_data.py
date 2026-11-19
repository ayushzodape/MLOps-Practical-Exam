import pandas as pd
import numpy as np

# Read original dataset
df = pd.read_csv("data/raw/wine.csv")

# Generate reproducible synthetic samples
rng = np.random.default_rng(42)

# Select 20 existing samples
sample = df.sample(20, random_state=42).copy()

# Add small noise to the 13 input features
noise = rng.normal(0, 0.05, size=(20, 13))

sample.iloc[:, :13] = (
    sample.iloc[:, :13].values + noise
)

# Combine original and synthetic samples
augmented = pd.concat(
    [df, sample],
    ignore_index=True
)

# Save updated dataset
augmented.to_csv(
    "data/raw/wine.csv",
    index=False
)

print("Wine dataset augmented successfully!")
print("Original rows: 178")
print("New rows:", len(augmented))