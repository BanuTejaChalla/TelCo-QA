
import re
import pandas as pd


def preprocess_dataset(df):
    """
    Prepare dataset text for quality evaluation.

    Preserves original text and creates separate cleaned
    and normalized text columns.

    Original dataset records are never deleted.
    """

    processed_df = df.copy()

    # Preserve original text
    processed_df["original_text"] = processed_df["text"]

    # Handle missing text without deleting records
    processed_df["cleaned_text"] = (
        processed_df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Normalize text for duplicate detection
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    processed_df["normalized_text"] = (
        processed_df["cleaned_text"]
        .apply(normalize_text)
    )

    # Flag empty records
    processed_df["is_empty"] = (
        processed_df["cleaned_text"] == ""
    )

    return processed_df
