import os
from sklearn.datasets import load_digits

os.makedirs("data/raw", exist_ok=True)
df = load_digits(as_frame=True).frame
df.to_csv("data/raw/dataset.csv", index=False)
print("Saved", df.shape, "-> data/raw/dataset.csv")
