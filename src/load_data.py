import os
from sklearn.datasets import load_wine            # Aim 1
# from sklearn.datasets import load_breast_cancer # Aim 2
# from sklearn.datasets import load_digits        # Aim 3

os.makedirs("data/raw", exist_ok=True)
df = load_wine(as_frame=True).frame               # swap the loader to match your aim
df.to_csv("data/raw/dataset.csv", index=False)
print("Saved", df.shape, "-> data/raw/dataset.csv")
