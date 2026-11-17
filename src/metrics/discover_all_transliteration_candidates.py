
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
import re

import pandas as pd


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_all_candidates.csv"
)

MIN_COUNT = 5
MIN_WORD_LENGTH = 3
MIN_SIMILARITY = 0.78
MAX_LENGTH_DIFFERENCE = 2
MAX_CONTEXTS = 2


# --------------------------------------------------
# WORD NORMALIZATION (DISCOVERY ONLY)
# --------------------------------------------------

def collapse_repeated_vowels(word):
    return re.sub(r"([aeiou])\1+", r"\1", word)


def transliteration_normalize(word):
    """
    Apply broad spelling simplifications for candidate discovery.

    These transformations are NOT applied to the original dataset.
    Matching words are only candidate suggestions.
    """
    word = collapse_repeated_vowels(word)
    word = word.replace("dh", "d")
    word = word.replace("th", "t")
    return word


# --------------------------------------------------
# DATA EXTRACTION
# --------------------------------------------------

def extract_word_data(df):
    eligible_df = df[
        df["script"].astype(str).str.strip().str.lower().isin(
            ["romanized", "code-mixed"]
        )
    ]

    if "cleaned_text" not in eligible_df.columns:
        raise ValueError("Dataset must contain cleaned_text.")

    word_counts = Counter()
    word_contexts = defaultdict(list)

    for _, record in eligible_df.iterrows():
        sentence = str(record["cleaned_text"]).lower()

        if not sentence or sentence == "nan":
            continue

        words = re.findall(r"\b[a-z]+\b", sentence)
        word_counts.update(words)

        for word in set(words):
            if len(word_contexts[word]) < MAX_CONTEXTS:
                original_text = (
                    record["text"]
                    if "text" in record.index
                    else record["cleaned_text"]
                )

                word_contexts[word].append(str(original_text))

    return eligible_df, word_counts, word_contexts


# --------------------------------------------------
# CANDIDATE CLASSIFICATION
# --------------------------------------------------

def get_candidate_reason(word_1, word_2, similarity):
    collapsed_1 = collapse_repeated_vowels(word_1)
    collapsed_2 = collapse_repeated_vowels(word_2)

    if collapsed_1 == collapsed_2:
        return "repeated_vowel"

    normalized_1 = transliteration_normalize(word_1)
    normalized_2 = transliteration_normalize(word_2)

    if normalized_1 == normalized_2:
        return "transliteration_pattern"

    return "general_similarity"


# --------------------------------------------------
# CANDIDATE DISCOVERY
# --------------------------------------------------

def discover_candidates(word_counts, word_contexts):
    eligible_words = sorted(
        word
        for word, count in word_counts.items()
        if count >= MIN_COUNT and len(word) >= MIN_WORD_LENGTH
    )

    candidates = []

    # Compare all eligible unique-word pairs.
    for word_1, word_2 in combinations(eligible_words, 2):

        if abs(len(word_1) - len(word_2)) > MAX_LENGTH_DIFFERENCE:
            continue

        similarity = SequenceMatcher(
            None, word_1, word_2
        ).ratio()

        normalized_1 = transliteration_normalize(word_1)
        normalized_2 = transliteration_normalize(word_2)

        same_normalized_form = normalized_1 == normalized_2

        if similarity < MIN_SIMILARITY and not same_normalized_form:
            continue

        reason = get_candidate_reason(
            word_1, word_2, similarity
        )

        candidates.append({
            "word_1": word_1,
            "word_2": word_2,
            "frequency_1": word_counts[word_1],
            "frequency_2": word_counts[word_2],
            "total_frequency": (
                word_counts[word_1] + word_counts[word_2]
            ),
            "similarity": round(similarity, 4),
            "candidate_reason": reason,
            "context_1": " || ".join(word_contexts[word_1]),
            "context_2": " || ".join(word_contexts[word_2]),
            "annotation": "REVIEW"
        })

    candidates.sort(
        key=lambda item: (
            item["total_frequency"],
            item["similarity"]
        ),
        reverse=True
    )

    return candidates, eligible_words


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    from src.data_loader import load_dataset
    from src.preprocessing import preprocess_dataset

    print("=" * 65)
    print("DATASET-WIDE TRANSLITERATION CANDIDATE DISCOVERY")
    print("=" * 65)

    df = load_dataset()
    processed_df = preprocess_dataset(df)

    eligible_df, word_counts, word_contexts = extract_word_data(
        processed_df
    )

    print(f"\nTotal dataset records: {len(processed_df)}")
    print(f"Romanized/code-mixed records: {len(eligible_df)}")
    print(f"Unique Latin-script words: {len(word_counts)}")

    candidates, eligible_words = discover_candidates(
        word_counts,
        word_contexts
    )

    print(f"Words considered (frequency >= {MIN_COUNT}): "
          f"{len(eligible_words)}")

    print(f"Candidate pairs discovered: {len(candidates)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.DataFrame(candidates)
    candidate_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved candidate report:")
    print(OUTPUT_PATH)

    if not candidate_df.empty:
        print("\nCandidate reason distribution:")
        print(candidate_df["candidate_reason"].value_counts())

        print("\nTop 20 candidates:")
        print(
            candidate_df[
                [
                    "word_1",
                    "word_2",
                    "frequency_1",
                    "frequency_2",
                    "similarity",
                    "candidate_reason"
                ]
            ].head(20).to_string(index=False)
        )

    print("\nIMPORTANT:")
    print("- These are candidate pairs, not confirmed equivalents.")
    print("- REVIEW labels are placeholders for human validation.")
    print("- Original dataset text and annotations were not modified.")
