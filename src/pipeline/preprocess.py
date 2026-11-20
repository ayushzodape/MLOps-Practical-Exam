# src/pipeline/preprocess.py
"""
Stage 1: Data Preprocessing.
Removes duplicates, fixes types, fills missing pixel values with the median
and drops rows with no label.
"""
import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    initial_rows = len(df)

    # Drop exact duplicate records
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", initial_rows - len(df))

    # Convert pixel columns to numbers and fill missing values with the median
    pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
    for col in pixel_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed %d missing values in '%s' with median=%.3f", n_missing, col, median_val)

    # Drop rows with an unresolvable missing target, then fix its type
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Preprocessed %d rows -> %s", len(df), output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/dataset.csv")
    parser.add_argument("--output", default="data/processed/digits_preprocessed.csv")
    args = parser.parse_args()
    preprocess(args.input, args.output)