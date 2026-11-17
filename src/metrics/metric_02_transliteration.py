
from collections import Counter, defaultdict
from pathlib import Path
import re

import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANLAB_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_label_issue_report.csv"
)


# --------------------------------------------------
# DATA SELECTION
# --------------------------------------------------

def get_latin_records(df):
    """Select Romanized and Code-mixed records."""

    return df[
        df["script"].astype(str).str.strip().str.lower().isin(
            ["romanized", "code-mixed"]
        )
    ]


# --------------------------------------------------
# FREQUENT WORD ANALYSIS
# --------------------------------------------------

def inspect_frequent_words(df, top_n=50):
    """Display frequent words in Latin-script records."""

    eligible_df = get_latin_records(df)
    word_counter = Counter()

    for text in eligible_df["cleaned_text"]:
        if not text:
            continue

        words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
        word_counter.update(words)

    print("\n" + "=" * 55)
    print("       FREQUENT WORDS IN LATIN-SCRIPT RECORDS")
    print("=" * 55)

    print(f"Records inspected: {len(eligible_df)}")
    print(f"Unique words: {len(word_counter)}")

    print("\nTop frequent words:")
    print("-" * 35)

    for word, count in word_counter.most_common(top_n):
        print(f"{word:<25} {count:>6}")


# --------------------------------------------------
# CANDIDATE VARIANT ANALYSIS
# --------------------------------------------------

def inspect_candidate_variants(df):
    """Count occurrences of manually identified candidate variants."""

    candidate_groups = {
        "idhi / idi": ["idhi", "idi"],
        "na / naa": ["na", "naa"],
        "chala / chaala": ["chala", "chaala"],
        "bagundi / baagundi": ["bagundi", "baagundi"]
    }

    all_text = " ".join(
        get_latin_records(df)["cleaned_text"]
        .dropna()
        .astype(str)
    ).lower()

    print("\n" + "=" * 55)
    print("       CANDIDATE TRANSLITERATION VARIANTS")
    print("=" * 55)

    for group, variants in candidate_groups.items():

        print(f"\n{group}")
        print("-" * 35)

        for word in variants:
            pattern = rf"\b{re.escape(word)}\b"
            count = len(re.findall(pattern, all_text))

            print(f"{word:<20} {count:>6}")


# --------------------------------------------------
# EXISTING VARIANT CONTEXT INSPECTION
# --------------------------------------------------

def inspect_variant_contexts(df, max_examples=5):
    """Display example sentences for manually identified variants."""

    candidate_groups = {
        "idhi / idi": ["idhi", "idi"],
        "na / naa": ["na", "naa"],
        "chala / chaala": ["chala", "chaala"],
        "bagundi / baagundi": ["bagundi", "baagundi"]
    }

    eligible_df = get_latin_records(df)

    print("\n" + "=" * 55)
    print("       TRANSLITERATION VARIANT CONTEXTS")
    print("=" * 55)

    for group, variants in candidate_groups.items():

        print(f"\n{group}")
        print("-" * 45)

        for word in variants:

            pattern = rf"\b{re.escape(word)}\b"

            matches = eligible_df[
                eligible_df["cleaned_text"].str.contains(
                    pattern,
                    case=False,
                    regex=True,
                    na=False
                )
            ]

            print(f"\n{word} - {len(matches)} records")

            for sentence in matches["cleaned_text"].head(max_examples):
                print(f"  - {sentence}")


# --------------------------------------------------
# CLEANLAB FLAGGED PAIR CONTEXT INSPECTION
# --------------------------------------------------

def inspect_flagged_pair_contexts(
    df,
    report_path=CLEANLAB_REPORT,
    max_examples=2
):
    """
    Retrieve dataset sentence contexts for every Cleanlab-flagged pair.

    Original annotations and dataset records are not modified.
    """

    if not report_path.exists():
        raise FileNotFoundError(
            f"Cleanlab report not found: {report_path}"
        )

    required_columns = [
        "word_1",
        "word_2",
        "label",
        "suggested_label",
        "assigned_label_probability",
        "cleanlab_flagged"
    ]

    report = pd.read_csv(report_path)

    missing = [
        column for column in required_columns
        if column not in report.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in Cleanlab report: {missing}"
        )

    # Handle Boolean values loaded from CSV.
    flagged_mask = (
        report["cleanlab_flagged"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )

    flagged_pairs = report[flagged_mask].copy()

    eligible_df = get_latin_records(df)

    if "cleaned_text" not in eligible_df.columns:
        raise ValueError(
            "The dataset must contain a cleaned_text column."
        )

    # Use original text for display whenever available.
    display_column = (
        "text" if "text" in eligible_df.columns
        else "cleaned_text"
    )

    print("\n" + "=" * 70)
    print("       CONTEXT INSPECTION OF CLEANLAB-FLAGGED PAIRS")
    print("=" * 70)

    print(f"Flagged pairs: {len(flagged_pairs)}")
    print(f"Dataset records searched: {len(eligible_df)}")

    for pair_number, (_, pair) in enumerate(
        flagged_pairs.iterrows(),
        start=1
    ):

        word_1 = str(pair["word_1"]).strip()
        word_2 = str(pair["word_2"]).strip()

        print("\n" + "=" * 70)
        print(f"PAIR {pair_number}/{len(flagged_pairs)}")
        print(f"{word_1}  <->  {word_2}")
        print("-" * 70)

        print(
            f"Original label : {int(pair['label'])} "
            f"({ 'Same' if int(pair['label']) == 1 else 'Different' })"
        )

        print(
            f"Model suggestion: {int(pair['suggested_label'])} "
            f"({ 'Same' if int(pair['suggested_label']) == 1 else 'Different' })"
        )

        print(
            "Confidence in original label: "
            f"{pair['assigned_label_probability']:.4f}"
        )

        for word in [word_1, word_2]:

            pattern = rf"\b{re.escape(word)}\b"

            matches = eligible_df[
                eligible_df["cleaned_text"].str.contains(
                    pattern,
                    case=False,
                    regex=True,
                    na=False
                )
            ]

            print(f"\nContexts for '{word}': {len(matches)} records")
            print("-" * 50)

            if matches.empty:
                print("No matching sentence found.")
                continue

            for _, record in matches.head(max_examples).iterrows():

                if "id" in record.index:
                    print(f"Record ID: {record['id']}")

                print(f"Sentence: {record[display_column]}")

    print("\n" + "=" * 70)
    print("CONTEXT INSPECTION COMPLETED")
    print("No dataset records or annotations were modified.")
    print("=" * 70)


# --------------------------------------------------
# AUTOMATIC VOWEL VARIANT DISCOVERY
# --------------------------------------------------

def discover_vowel_variants(df, min_count=5):
    """
    Discover possible spelling variants formed by repeated vowels.

    These are candidate suggestions only.
    They must be checked in context before being treated as equivalent.
    """

    eligible_df = get_latin_records(df)
    word_counter = Counter()

    for text in eligible_df["cleaned_text"].dropna():
        words = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
        word_counter.update(words)

    def collapse_repeated_vowels(word):
        return re.sub(r"([aeiou])\1+", r"\1", word)

    variant_groups = defaultdict(list)

    for word, count in word_counter.items():

        if len(word) < 2 or count < min_count:
            continue

        normalized_word = collapse_repeated_vowels(word)
        variant_groups[normalized_word].append((word, count))

    print("\n" + "=" * 55)
    print("       AUTOMATIC VARIANT CANDIDATE DISCOVERY")
    print("=" * 55)

    suggestions = []

    for normalized_word, variants in variant_groups.items():

        if len(variants) < 2:
            continue

        variants.sort(key=lambda item: item[1], reverse=True)
        suggestions.append((normalized_word, variants))

    suggestions.sort(
        key=lambda item: sum(count for _, count in item[1]),
        reverse=True
    )

    print(f"Candidate groups discovered: {len(suggestions)}")
    print("\nPossible repeated-vowel variations:")
    print("-" * 55)

    for normalized_word, variants in suggestions:

        variant_text = ", ".join(
            f"{word} ({count})"
            for word, count in variants
        )

        print(f"{normalized_word:<18} {variant_text}")

    print("\nNote: These are unverified candidates, not confirmed equivalents.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    from src.data_loader import load_dataset
    from src.preprocessing import preprocess_dataset

    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: Preprocess dataset
    processed_df = preprocess_dataset(df)

    # Step 3: Inspect frequent words
    inspect_frequent_words(processed_df)

    # Step 4: Inspect existing candidate variants
    inspect_candidate_variants(processed_df)

    # Step 5: Inspect existing variant contexts
    inspect_variant_contexts(processed_df)

    # Step 6: Discover additional possible variants
    discover_vowel_variants(processed_df)

    # Step 7: Inspect all Cleanlab-flagged pairs
    inspect_flagged_pair_contexts(
        processed_df,
        max_examples=2
    )
