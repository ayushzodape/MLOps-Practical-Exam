# src/pipeline/split.py
"""
Stage 4: Scale and split.
Splits the validated features into train/test and scales them
(scaler is fitted on the training set only, to avoid data leakage).
"""
import argparse
import logging
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("split")


def split(input_path: str, train_path: str, test_path: str) -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["split"]

    df = pd.read_csv(input_path)
    X = df.drop(columns=["target"])
    y = df["target"]

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

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info("Saved train %s -> %s and test %s -> %s", train.shape, train_path, test.shape, test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/digits_features.csv")
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    args = parser.parse_args()
    split(args.input, args.train, args.test)