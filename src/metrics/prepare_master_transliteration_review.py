
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CANDIDATES_PATH = DATA_DIR / "transliteration_all_candidates.csv"
ANNOTATIONS_PATH = DATA_DIR / "transliteration_annotations_all.csv"

OUTPUT_PATH = DATA_DIR / "transliteration_master_review.csv"


def normalize_pair(word_1, word_2):
    return tuple(sorted([
        str(word_1).strip().lower(),
        str(word_2).strip().lower()
    ]))


def main():
    print("\n" + "=" * 65)
    print("       TELCO-QA | MASTER TRANSLITERATION REVIEW")
    print("=" * 65)

    candidates = pd.read_csv(CANDIDATES_PATH)
    annotations = pd.read_csv(ANNOTATIONS_PATH)

    required_candidate_columns = [
        "word_1",
        "word_2",
        "frequency_1",
        "frequency_2",
        "similarity",
        "candidate_reason",
        "context_1",
        "context_2"
    ]

    missing = [
        column for column in required_candidate_columns
        if column not in candidates.columns
    ]

    if missing:
        raise ValueError(
            f"Candidate file is missing columns: {missing}"
        )

    known_pairs = {
        normalize_pair(row.word_1, row.word_2)
        for _, row in annotations.iterrows()
    }

    candidates["pair_id"] = candidates.apply(
        lambda row: normalize_pair(row["word_1"], row["word_2"]),
        axis=1
    )

    # Exclude all pairs already reviewed.
    review = candidates[
        ~candidates["pair_id"].isin(known_pairs)
    ].copy()

    # Remove duplicate candidate pairs, if any.
    review = review.drop_duplicates(
        subset="pair_id"
    )

    review["total_frequency"] = (
        review["frequency_1"] + review["frequency_2"]
    )

    review = review.sort_values(
        by=["total_frequency", "similarity"],
        ascending=False
    )

    # Add blank fields for your judgments.
    review["annotation"] = ""
    review["review_notes"] = ""

    output_columns = [
        "word_1",
        "word_2",
        "frequency_1",
        "frequency_2",
        "total_frequency",
        "similarity",
        "candidate_reason",
        "context_1",
        "context_2",
        "annotation",
        "review_notes"
    ]

    review = review[output_columns]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    review.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nExisting annotations: {len(annotations)}")
    print(f"Original candidates: {len(candidates)}")
    print(f"Already reviewed: {len(candidates) - len(review)}")
    print(f"New pairs for review: {len(review)}")

    print(f"\nMaster review file saved:\n{OUTPUT_PATH}")

    print("\nAnnotation labels:")
    print("1       = Same underlying Telugu word")
    print("0       = Different Telugu word or form")
    print("ENGLISH = English word/control")
    print("REVIEW  = Cannot confidently determine")

    print("\nNo existing annotations or dataset records were modified.")

    print("\n" + "=" * 65)
    print("       MASTER REVIEW PREPARATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
