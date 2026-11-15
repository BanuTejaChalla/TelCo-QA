
from pathlib import Path
from collections import Counter
from itertools import combinations
import re

import pandas as pd
from rapidfuzz.fuzz import ratio

from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_candidate_pairs.csv"
)


def extract_word_frequencies(df):
    """Extract words from Romanized and Code-mixed records."""

    eligible_df = df[
        df["script"].astype(str).str.strip().str.lower().isin(
            ["romanized", "code-mixed"]
        )
    ]

    word_counter = Counter()

    for text in eligible_df["cleaned_text"].dropna():
        words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
        word_counter.update(words)

    return word_counter


def normalize_transliteration(word):
    """
    Apply conservative transformations to discover
    possible Romanized Telugu spelling variations.

    These transformations generate candidates only.
    They do not establish linguistic equivalence.
    """

    word = word.lower()

    # Collapse repeated vowels
    word = re.sub(r"([aeiou])\1+", r"\1", word)

    # Common Romanized Telugu consonant variations
    word = word.replace("dh", "d")
    word = word.replace("th", "t")

    return word


def detect_english_inflection(word_1, word_2):
    """
    Flag possible English singular/plural variations.

    This is only a heuristic. It does not identify
    language with certainty.
    """

    if word_1 + "s" == word_2 or word_2 + "s" == word_1:
        return True

    if word_1 + "es" == word_2 or word_2 + "es" == word_1:
        return True

    return False


def get_word_contexts(df, word, max_examples=3):

    eligible_df = df[
        df["script"].astype(str).str.strip().str.lower().isin(
            ["romanized", "code-mixed"]
        )
    ]

    pattern = rf"\b{re.escape(word)}\b"

    matches = eligible_df[
        eligible_df["cleaned_text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ]

    examples = matches["cleaned_text"].head(max_examples).tolist()

    return " || ".join(examples)


def generate_candidate_pairs(word_counter, df, min_frequency=5):

    words = [
        word
        for word, count in word_counter.items()
        if count >= min_frequency and len(word) >= 2
    ]

    candidates = []

    for word_1, word_2 in combinations(sorted(words), 2):

        if abs(len(word_1) - len(word_2)) > 3:
            continue

        similarity = ratio(word_1, word_2)

        normalized_1 = normalize_transliteration(word_1)
        normalized_2 = normalize_transliteration(word_2)

        reasons = []

        if normalized_1 == normalized_2:
            reasons.append("Possible Telugu transliteration pattern")

        if similarity >= 85:
            reasons.append("High character similarity")

        if not reasons:
            continue

        possible_english = detect_english_inflection(
            word_1,
            word_2
        )

        if possible_english:
            candidate_type = "Possible English inflection"

        elif normalized_1 == normalized_2:
            candidate_type = "Transliteration pattern"

        else:
            candidate_type = "General similarity candidate"

        candidates.append({
            "word_1": word_1,
            "word_2": word_2,
            "frequency_1": word_counter[word_1],
            "frequency_2": word_counter[word_2],
            "similarity": round(similarity, 2),
            "candidate_type": candidate_type,
            "candidate_reason": "; ".join(reasons),
            "context_1": get_word_contexts(df, word_1),
            "context_2": get_word_contexts(df, word_2),
            "label": "",
            "notes": ""
        })

    # Prioritize transliteration patterns.
    # Keep English controls and other candidates for later review.
    priority = {
        "Transliteration pattern": 0,
        "General similarity candidate": 1,
        "Possible English inflection": 2
    }

    candidates.sort(
        key=lambda item: (
            priority[item["candidate_type"]],
            -item["similarity"],
            -min(item["frequency_1"], item["frequency_2"])
        )
    )

    return candidates


def main():

    print("\n" + "=" * 60)
    print("       TELCO-QA | TRANSLITERATION CANDIDATE DISCOVERY")
    print("=" * 60)

    print("\n[1] Loading dataset...")

    df = load_dataset()
    processed_df = preprocess_dataset(df)

    print(f"    Records loaded: {len(processed_df)}")

    print("\n[2] Extracting word frequencies...")

    word_counter = extract_word_frequencies(processed_df)

    print(f"    Unique words found: {len(word_counter)}")

    print("\n[3] Discovering candidate pairs...")

    candidates = generate_candidate_pairs(
        word_counter,
        processed_df
    )

    print(f"    Candidate pairs generated: {len(candidates)}")

    print("\n[4] Saving candidate pairs...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.DataFrame(candidates)

    candidate_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"    Saved to: {OUTPUT_PATH}")

    print("\n" + "-" * 60)
    print("              CANDIDATE CATEGORY SUMMARY")
    print("-" * 60)

    if not candidate_df.empty:
        print(
            candidate_df["candidate_type"]
            .value_counts()
            .to_string()
        )

        print("\nPrioritized candidate preview:")

        print(
            candidate_df[
                [
                    "word_1",
                    "word_2",
                    "similarity",
                    "candidate_type"
                ]
            ].head(20).to_string(index=False)
        )

    print("\n" + "=" * 60)
    print("       CANDIDATE DISCOVERY COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
