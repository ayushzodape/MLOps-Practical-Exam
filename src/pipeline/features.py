# src/pipeline/features.py
"""
Stage 2: Feature Engineering.
Derives numeric features from the 8x8 pixel grid (ink amount and where the ink is).
"""
import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("features")


def engineer_features(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    pixel_cols = [c for c in df.columns if c.startswith("pixel_")]

    # Column names look like pixel_<row>_<col>
    def cols_where(test):
        return [c for c in pixel_cols if test(*map(int, c.split("_")[1:]))]

    top = cols_where(lambda r, c: r < 4)
    bottom = cols_where(lambda r, c: r >= 4)
    left = cols_where(lambda r, c: c < 4)
    right = cols_where(lambda r, c: c >= 4)

    # Creating new features
    df["pixel_sum"] = df[pixel_cols].sum(axis=1)                 # total ink
    df["nonzero_pixels"] = (df[pixel_cols] > 0).sum(axis=1)      # how many pixels are inked
    df["top_half_sum"] = df[top].sum(axis=1)
    df["bottom_half_sum"] = df[bottom].sum(axis=1)
    df["left_half_sum"] = df[left].sum(axis=1)
    df["right_half_sum"] = df[right].sum(axis=1)

    # Keep the label as the last column
    cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df[cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Engineered %d columns -> %s", df.shape[1], output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/digits_preprocessed.csv")
    parser.add_argument("--output", default="data/processed/digits_features.csv")
    args = parser.parse_args()
    engineer_features(args.input, args.output)