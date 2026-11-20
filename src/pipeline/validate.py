# src/pipeline/validate.py
"""
Stage 3: Data Validation.
Schema and statistical checks that must pass before data can flow downstream.
Raises on failure to hard-stop the pipeline (exit code 1).
"""
import argparse
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")

PIXEL_COLS = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]
ENGINEERED_COLS = [
    "pixel_sum", "nonzero_pixels",
    "top_half_sum", "bottom_half_sum", "left_half_sum", "right_half_sum",
]
EXPECTED_COLUMNS = set(PIXEL_COLS + ENGINEERED_COLS + ["target"])
VALID_TARGETS = set(range(10))
RANGE_CHECKS = {col: (0, 16) for col in PIXEL_COLS}
RANGE_CHECKS.update({
    "pixel_sum": (0, 1024),
    "nonzero_pixels": (0, 64),
    "top_half_sum": (0, 512),
    "bottom_half_sum": (0, 512),
    "left_half_sum": (0, 512),
    "right_half_sum": (0, 512),
})
MIN_CLASS_SHARE = 0.05   # every digit should be at least 5% of the data


class DataValidationError(Exception):
    pass


def validate(input_path: str, report_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    errors = []

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing expected columns: {sorted(missing_cols)}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Unexpected null values in columns: {null_cols}")

    if "target" in df.columns:
        invalid = set(df["target"].dropna().unique()) - VALID_TARGETS
        if invalid:
            errors.append(f"Unexpected target values: {invalid}")
        shares = df["target"].value_counts(normalize=True)
        rare = shares[shares < MIN_CLASS_SHARE]
        if not rare.empty:
            errors.append(f"Classes under {MIN_CLASS_SHARE:.0%} of the data: {rare.round(3).to_dict()}")

    for col, (low, high) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            errors.append(f"{len(out_of_range)} rows out of expected range for '{col}' ({low}-{high})")

    if errors:
        for e in errors:
            logger.error(e)
        raise DataValidationError(f"Validation failed with {len(errors)} error(s)")

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report = {
            "status": "PASSED",
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "class_counts": {str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()},
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    logger.info("Validation PASSED: %d rows, %d columns, all checks satisfied", len(df), df.shape[1])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/digits_features.csv")
    parser.add_argument("--report", default="data/validated/validation_report.json")
    args = parser.parse_args()
    try:
        validate(args.input, args.report)
    except DataValidationError as e:
        logger.error("Pipeline halted: %s", e)
        sys.exit(1)   # non-zero exit code = stage failed