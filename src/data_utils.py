"""
data_utils.py
=============
Shared data-loading, feature-engineering, and text-verbalization utilities
used by all three members' pipelines (HimasNLP, AfrithNLP, IlmaanNLP).

Dataset actually used: Instagram.csv / Instagram_fake_profile_dataset.csv
(Instagram Fake Spammer Genuine Accounts Dataset, Kaggle)

IMPORTANT NOTE ON THE DATASET
------------------------------
The real dataset provided for this project is fully STRUCTURED / NUMERIC.
It does NOT contain a raw biography text string. Its columns are:

    profile pic            -> 1 if the account has a profile picture, else 0
    nums/length username   -> ratio of digits to length in the username
    fullname words         -> number of words in the full name
    nums/length fullname   -> ratio of digits to length in the full name
    name==username         -> 1 if full name equals username
    description length     -> character length of the bio/description
    external URL            -> 1 if the bio contains an external link
    private                 -> 1 if the account is private
    #posts                  -> number of posts
    #followers               -> number of followers
    #follows                 -> number of accounts followed
    followers_following_ratio (Instagram.csv only) -> engineered ratio feature
    fake                     -> target label (0 = real, 1 = fake)

Since the project brief (Section 3) requires BoW / TF-IDF / BERT-based NLP
feature extraction on "biography text", and no raw bio string is available,
this module VERBALIZES each row's structured profile signals into a short
natural-language profile description (a standard technique sometimes called
"data-to-text" or "feature verbalization"). This gives every downstream NLP
model (BoW, TF-IDF, LSTM, CNN, BERT) genuine text to tokenize, vectorize, or
embed, while being fully transparent that the text is derived from
structured fields rather than a scraped bio string.

If you later obtain a version of the dataset with an actual free-text bio
column (e.g. named 'bio', 'biography', 'description'), this module will
automatically detect and use it instead of the verbalized text.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Candidate column names that might hold real free-text biography content
POSSIBLE_TEXT_COLUMNS = ["bio", "biography", "bio_text", "profile_bio"]

# Numeric/categorical columns from the Kaggle dataset (superset -- code
# below only uses the ones actually present in the loaded CSV)
NUMERIC_COLUMNS = [
    "profile pic",
    "nums/length username",
    "fullname words",
    "nums/length fullname",
    "name==username",
    "description length",
    "external URL",
    "private",
    "#posts",
    "#followers",
    "#follows",
    "followers_following_ratio",
]

TARGET_COLUMN = "fake"


def load_raw(split="train"):
    """Load train.csv or test.csv from the data/ directory."""
    path = os.path.join(DATA_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Place train.csv / test.csv inside the "
            f"data/ folder (see README for how the split was generated)."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_text_column(df):
    """Return the name of a real free-text bio column if one exists."""
    for col in POSSIBLE_TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None


def get_numeric_features(df):
    """Return only the numeric/categorical profile features that exist."""
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    return df[cols].copy()


def get_target(df):
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    return df[TARGET_COLUMN].astype(int)


def verbalize_row(row: pd.Series) -> str:
    """
    Convert one row of structured Instagram profile features into a short
    natural-language "profile description" so text-based NLP models
    (BoW / TF-IDF / LSTM / CNN / BERT) have real text to process.
    """
    parts = []

    parts.append("has profile picture" if row.get("profile pic", 0) == 1
                 else "no profile picture")

    uname_ratio = row.get("nums/length username", 0)
    if uname_ratio == 0:
        parts.append("username has no digits")
    elif uname_ratio < 0.2:
        parts.append("username has few digits")
    else:
        parts.append("username has many digits")

    fullname_words = int(row.get("fullname words", 0))
    parts.append(f"full name contains {fullname_words} words")

    if row.get("name==username", 0) == 1:
        parts.append("full name matches username")

    desc_len = row.get("description length", 0)
    if desc_len == 0:
        parts.append("empty biography")
    elif desc_len < 20:
        parts.append("very short biography")
    elif desc_len < 60:
        parts.append("short biography")
    else:
        parts.append("long biography")

    parts.append("bio contains external link" if row.get("external URL", 0) == 1
                 else "no external link in bio")

    parts.append("private account" if row.get("private", 0) == 1
                 else "public account")

    posts = row.get("#posts", 0)
    if posts == 0:
        parts.append("zero posts")
    elif posts < 10:
        parts.append("very few posts")
    elif posts < 100:
        parts.append("moderate number of posts")
    else:
        parts.append("many posts")

    followers = row.get("#followers", 0)
    follows = row.get("#follows", 0)
    parts.append(f"{int(followers)} followers")
    parts.append(f"following {int(follows)} accounts")

    if follows > 0:
        ratio = followers / follows
        if ratio < 0.5:
            parts.append("follows far more accounts than followers suspicious ratio")
        elif ratio > 5:
            parts.append("many more followers than accounts followed influencer like ratio")
        else:
            parts.append("balanced follower to following ratio")

    return ", ".join(parts)


def build_text_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a text Series for NLP processing: a real bio column if present,
    otherwise a verbalized description built from structured features.
    """
    text_col = detect_text_column(df)
    if text_col:
        return df[text_col].fillna("").astype(str)
    return df.apply(verbalize_row, axis=1)


def load_split(split="train"):
    """
    Convenience loader used by every member's pipeline.

    Returns
    -------
    numeric_df  : pd.DataFrame -- numeric/categorical profile features
    text_series : pd.Series    -- real bio text, or verbalized profile text
    y           : pd.Series    -- target labels (0 = real, 1 = fake)
    """
    df = load_raw(split)
    text_series = build_text_series(df)
    numeric_df = get_numeric_features(df)
    y = get_target(df)
    return numeric_df, text_series, y


if __name__ == "__main__":
    numeric_df, text_series, y = load_split("train")
    print("Numeric feature shape:", numeric_df.shape)
    print("Sample verbalized text:\n", text_series.iloc[0])
    print("\nTarget distribution:\n", y.value_counts())
