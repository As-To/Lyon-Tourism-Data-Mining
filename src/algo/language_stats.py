"""
Utilities to quantify multilingual content in the dataset.

Given a DataFrame with ``tags`` and ``title`` columns, this module:
- builds a combined text column (tags + title) with NaN handled as empty
- applies a light normalization (lowercase, remove digits/special chars, normalize spaces)
- detects rows containing non-Latin characters (Arabic, Cyrillic, CJK)
- estimates how many tokens are outside French/English stopword lists
"""

from __future__ import annotations

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Ensure French stopwords are available.
nltk.download("stopwords", quiet=True)

# Unicode ranges for Arabic, Cyrillic, Japanese (Hiragana/Katakana), and CJK.
NON_LATIN_REGEX = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF]")

# Keep Latin (with accents) + the non-Latin ranges above; drop everything else.
CLEANUP_REGEX = re.compile(r"[^a-z\u00c0-\u024f\u0400-\u04ff\u0600-\u06ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\s]")

FRENCH_STOPWORDS = set(stopwords.words("french"))
DEFAULT_STOPWORDS = set(ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """
    Concatenate tags and title into a single text column.
    Missing values are replaced by empty strings.
    """
    tags = df["tags"] if "tags" in df.columns else pd.Series("", index=df.index)
    title = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    return (tags.fillna("") + " " + title.fillna("")).str.strip()


def basic_preprocess(text: str) -> str:
    """
    Lowercase, strip digits/special characters, and normalize whitespace.
    Keeps alphabetic characters across Latin, Arabic, Cyrillic, and CJK blocks.
    """
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = CLEANUP_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_non_latin(text: str) -> bool:
    """Return True if the text contains Arabic, Cyrillic, or CJK characters."""
    return bool(NON_LATIN_REGEX.search(str(text)))


def ratio_non_fr_en_tokens(text: str, stopwords_set=DEFAULT_STOPWORDS) -> float:
    """
    Compute the fraction of tokens that are not in French or English stopword lists.
    Returns 0.0 when no tokens are present.
    """
    tokens = [tok for tok in str(text).split() if tok]
    if not tokens:
        return 0.0
    non_stop = [tok for tok in tokens if tok not in stopwords_set]
    return len(non_stop) / len(tokens)


def compute_language_stats(df: pd.DataFrame) -> dict:
    """
    Main entry point.

    Returns a dictionary with:
    - percent_non_latin: percentage of rows containing non-Latin characters
    - mean_ratio_non_fr_en: mean ratio of tokens not in fr/en stopwords
    - median_ratio_non_fr_en: median ratio of tokens not in fr/en stopwords
    """
    if df is None or df.empty:
        return {
            "percent_non_latin": 0.0,
            "mean_ratio_non_fr_en": 0.0,
            "median_ratio_non_fr_en": 0.0,
        }

    combined = build_combined_text(df)

    non_latin_mask = combined.map(contains_non_latin)
    percent_non_latin = float(non_latin_mask.mean() * 100.0) if len(combined) else 0.0

    preprocessed = combined.map(basic_preprocess)
    ratios = preprocessed.map(lambda t: ratio_non_fr_en_tokens(t, DEFAULT_STOPWORDS))

    return {
        "percent_non_latin": percent_non_latin,
        "mean_ratio_non_fr_en": float(ratios.mean()) if not ratios.empty else 0.0,
        "median_ratio_non_fr_en": float(ratios.median()) if not ratios.empty else 0.0,
    }


def _demo_from_csv():
    """
    Quick manual check on the cleaned dataset.
    Adjust the path if you move the CSV.
    """
    csv_path = "data/cleaned/cleaned_lyon_data.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"CSV introuvable: {csv_path}")
        return

    stats = compute_language_stats(df)
    print("=== Language stats on cleaned_lyon_data.csv ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    _demo_from_csv()
