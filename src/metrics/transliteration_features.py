
from pathlib import Path
import re

import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_ml_training_data.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_feature_dataset.csv"
)

VOWELS = set("aeiou")


def clean_word(word):
    """Prepare a word for feature calculation without changing its spelling."""
    if pd.isna(word):
        return ""

    return str(word).strip().lower()


def common_prefix_length(word_1, word_2):
    """Count matching characters from the beginning."""
    count = 0

    for char_1, char_2 in zip(word_1, word_2):
        if char_1 != char_2:
            break
        count += 1

    return count


def common_suffix_length(word_1, word_2):
    """Count matching characters from the end."""
    count = 0

    for char_1, char_2 in zip(word_1[::-1], word_2[::-1]):
        if char_1 != char_2:
            break
        count += 1

    return count


def count_vowels(word):
    return sum(char in VOWELS for char in word)


def count_consonants(word):
    return sum(
        char.isalpha() and char not in VOWELS
        for char in word
    )


def count_repeated_vowels(word):
    """Count adjacent repeated vowels, such as aa, ee, or oo."""
    return sum(
        word[index] == word[index + 1]
        and word[index] in VOWELS
        for index in range(len(word) - 1)
    )


def extract_pair_features(word_1, word_2):
    """Extract interpretable numerical features from a word pair."""

    word_1 = clean_word(word_1)
    word_2 = clean_word(word_2)

    len_1 = len(word_1)
    len_2 = len(word_2)

    max_length = max(len_1, len_2)

    edit_distance = Levenshtein.distance(word_1, word_2)

    features = {
        "length_1": len_1,
        "length_2": len_2,

        "length_difference": abs(len_1 - len_2),

        "length_ratio": (
            min(len_1, len_2) / max_length
            if max_length > 0
            else 1.0
        ),

        "edit_distance": edit_distance,

        "normalized_edit_distance": (
            edit_distance / max_length
            if max_length > 0
            else 0.0
        ),

        "similarity_ratio": ratio(word_1, word_2) / 100,

        "common_prefix_length": common_prefix_length(
            word_1, word_2
        ),

        "common_suffix_length": common_suffix_length(
            word_1, word_2
        ),

        "vowel_count_1": count_vowels(word_1),
        "vowel_count_2": count_vowels(word_2),

        "vowel_count_difference": abs(
            count_vowels(word_1) - count_vowels(word_2)
        ),

        "consonant_count_1": count_consonants(word_1),
        "consonant_count_2": count_consonants(word_2),

        "consonant_count_difference": abs(
            count_consonants(word_1) - count_consonants(word_2)
        ),

        "repeated_vowels_1": count_repeated_vowels(word_1),
        "repeated_vowels_2": count_repeated_vowels(word_2),

        "repeated_vowel_difference": abs(
            count_repeated_vowels(word_1)
            - count_repeated_vowels(word_2)
        ),
    }

    return features


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"word_1", "word_2", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    feature_rows = []

    for _, row in df.iterrows():
        features = extract_pair_features(
            row["word_1"],
            row["word_2"]
        )

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)

    # Preserve the original annotations and metadata.
    result_df = pd.concat(
        [df.reset_index(drop=True), features_df],
        axis=1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nTRANSLITERATION FEATURE EXTRACTION")
    print("----------------------------------")

    print(f"Input records       : {len(df)}")
    print(f"Output records      : {len(result_df)}")
    print(f"Features extracted  : {len(features_df.columns)}")
    print(f"Missing feature values: {features_df.isna().sum().sum()}")

    print(f"\nOutput saved to:\n{OUTPUT_FILE}")

    print("\nFEATURE PREVIEW")
    print("----------------")

    preview_columns = [
        "word_1",
        "word_2",
        "label",
        "edit_distance",
        "normalized_edit_distance",
        "similarity_ratio",
        "vowel_count_difference",
        "consonant_count_difference",
        "repeated_vowel_difference",
    ]

    print(
        result_df[preview_columns].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
