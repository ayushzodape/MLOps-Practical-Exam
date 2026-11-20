import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

params = yaml.safe_load(open("params.yaml"))["preprocess"]

df = pd.read_csv("data/raw/dataset.csv")

# Clean
df = df.drop_duplicates().dropna()

X = df.drop(columns=["target"])
y = df["target"]

# Split first, then scale (fit on train only to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y,
)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

train = X_train.assign(target=y_train.values)
test = X_test.assign(target=y_test.values)

os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("train:", train.shape, "test:", test.shape)
