import os
import pandas as pd
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine(as_frame=True)

# Convert dataset into DataFrame
df = wine.frame

# Create directory if it does not exist
os.makedirs("data/raw", exist_ok=True)

# Save dataset as CSV
df.to_csv("data/raw/wine.csv", index=False)

print("Wine dataset generated successfully!")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("\nDataset preview:")
print(df.head())