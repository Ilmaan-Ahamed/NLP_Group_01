"""
make_train_test_split.py
=========================
Reproduces the train.csv / test.csv split used throughout this project from
the raw Kaggle dataset (data/Instagram_raw.csv).

80/20 stratified split on the 'fake' target, random_state=42.

Usage:
    python -m src.make_train_test_split
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    raw_path = os.path.join(DATA_DIR, "Instagram_raw.csv")
    df = pd.read_csv(raw_path)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["fake"]
    )

    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    print(f"train.csv: {train_df.shape}, test.csv: {test_df.shape}")
    print("Train class balance:\n", train_df["fake"].value_counts())
    print("Test class balance:\n", test_df["fake"].value_counts())


if __name__ == "__main__":
    main()
