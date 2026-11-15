
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "transliteration_candidate_pairs.csv"

PREVIOUS_BATCHES = [
    DATA_DIR / "transliteration_annotation_batch_01.csv",
    DATA_DIR / "transliteration_annotation_batch_02.csv",
    DATA_DIR / "transliteration_annotation_batch_03.csv"
]

OUTPUT_PATH = DATA_DIR / "transliteration_annotation_batch_04.csv"

BATCH_SIZE = 30


def normalize_pair(word_1, word_2):
    """Create an order-independent pair identifier."""

    return tuple(
        sorted([
            str(word_1).strip().lower(),
            str(word_2).strip().lower()
        ])
    )


def load_previous_pairs():

    previous_pairs = set()

    for batch_path in PREVIOUS_BATCHES:

        if not batch_path.exists():
            raise FileNotFoundError(
                f"Previous annotation batch not found: {batch_path}"
            )

        previous_batch = pd.read_csv(batch_path)

        for _, row in previous_batch.iterrows():

            previous_pairs.add(
                normalize_pair(row["word_1"], row["word_2"])
            )

    return previous_pairs


def create_annotation_batch():

    print("\n" + "=" * 60)
    print("       TELCO-QA | ANNOTATION BATCH 04")
    print("=" * 60)

    # Step 1: Load candidates
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Candidate pairs file not found. "
            "Generate candidate pairs first."
        )

    candidates = pd.read_csv(INPUT_PATH)

    # Step 2: Load all previously annotated pairs
    previous_pairs = load_previous_pairs()

    # Step 3: Create pair identifiers
    candidates["pair_id"] = candidates.apply(
        lambda row: normalize_pair(row["word_1"], row["word_2"]),
        axis=1
    )

    # Step 4: Exclude previously annotated pairs
    eligible = candidates[
        ~candidates["pair_id"].isin(previous_pairs)
    ].copy()

    # Step 5: Select next candidates
    batch = eligible.head(BATCH_SIZE).copy()

    if batch.empty:
        print("\nNo new candidate pairs available.")
        return

    # Step 6: Add annotation columns
    batch["annotation"] = ""
    batch["review_notes"] = ""

    columns = [
        "word_1",
        "word_2",
        "frequency_1",
        "frequency_2",
        "similarity",
        "candidate_type",
        "candidate_reason",
        "context_1",
        "context_2",
        "annotation",
        "review_notes"
    ]

    batch = batch[columns]

    # Step 7: Save Batch 04
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    batch.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nTotal candidates: {len(candidates)}")
    print(f"Previously annotated unique pairs: {len(previous_pairs)}")
    print(f"Remaining candidates: {len(eligible)}")
    print(f"Pairs selected: {len(batch)}")

    print(f"\nSaved to: {OUTPUT_PATH}")

    print("\nAnnotation labels:")
    print("1       = Same Telugu word")
    print("0       = Different Telugu word/form")
    print("ENGLISH = English control")
    print("REVIEW  = Ambiguous; needs further inspection")

    print("\nBatch 04 preview:")
    print("-" * 60)

    print(
        batch[
            [
                "word_1",
                "word_2",
                "similarity",
                "candidate_type"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("       BATCH 04 CREATED")
    print("=" * 60)


if __name__ == "__main__":
    create_annotation_batch()
