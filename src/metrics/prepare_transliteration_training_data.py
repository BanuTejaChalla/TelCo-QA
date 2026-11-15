
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

BATCH_FILES = [
    DATA_DIR / "transliteration_annotation_batch_01.csv",
    DATA_DIR / "transliteration_annotation_batch_02.csv",
    DATA_DIR / "transliteration_annotation_batch_03.csv",
    DATA_DIR / "transliteration_annotation_batch_04.csv",
]

ALL_ANNOTATIONS_PATH = (
    DATA_DIR / "transliteration_annotations_all.csv"
)

TRAINING_DATA_PATH = (
    DATA_DIR / "transliteration_ml_training_data.csv"
)

ENGLISH_CONTROLS_PATH = (
    DATA_DIR / "transliteration_english_controls.csv"
)

REVIEW_DATA_PATH = (
    DATA_DIR / "transliteration_review_pairs.csv"
)


def normalize_pair(word_1, word_2):
    """Create an order-independent pair identifier."""

    return tuple(
        sorted([
            str(word_1).strip().lower(),
            str(word_2).strip().lower()
        ])
    )


def normalize_annotation(value):
    """Standardize annotation labels."""

    value = str(value).strip().upper()

    if value in ["1", "1.0"]:
        return "1"

    if value in ["0", "0.0"]:
        return "0"

    if value == "ENGLISH":
        return "ENGLISH"

    if value == "REVIEW":
        return "REVIEW"

    raise ValueError(
        f"Unrecognized annotation label: {value!r}"
    )


def load_annotation_batches():

    all_batches = []

    for batch_number, batch_path in enumerate(
        BATCH_FILES, start=1
    ):

        print(f"\nLoading Batch {batch_number}...")

        if not batch_path.exists():
            raise FileNotFoundError(
                f"Annotation batch not found: {batch_path}"
            )

        batch = pd.read_csv(
            batch_path,
            dtype=str,
            keep_default_na=False
        )

        required_columns = [
            "word_1",
            "word_2",
            "annotation"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in batch.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Batch {batch_number} is missing columns: "
                f"{missing_columns}"
            )

        if batch["annotation"].str.strip().eq("").any():
            raise ValueError(
                f"Batch {batch_number} contains blank annotations. "
                "Complete them before consolidation."
            )

        batch["annotation"] = batch["annotation"].apply(
            normalize_annotation
        )

        batch["source_batch"] = f"Batch {batch_number:02d}"

        all_batches.append(batch)

        print(f"    Records loaded: {len(batch)}")

    return pd.concat(
        all_batches,
        ignore_index=True
    )


def validate_annotations(df):

    print("\nValidating annotations...")

    # Check for missing words
    missing_words = (
        df["word_1"].str.strip().eq("")
        | df["word_2"].str.strip().eq("")
    )

    if missing_words.any():
        raise ValueError(
            "Some annotation records contain empty words."
        )

    # Identify order-independent duplicate pairs
    df["pair_id"] = df.apply(
        lambda row: normalize_pair(
            row["word_1"],
            row["word_2"]
        ),
        axis=1
    )

    duplicate_mask = df["pair_id"].duplicated(keep=False)

    if duplicate_mask.any():

        duplicates = df.loc[
            duplicate_mask,
            [
                "word_1",
                "word_2",
                "annotation",
                "source_batch"
            ]
        ]

        raise ValueError(
            "Duplicate pairs found across annotation batches.\n"
            "Inspect these records before continuing:\n"
            + duplicates.to_string(index=False)
        )

    print("    Empty word check: Passed")
    print("    Duplicate pair check: Passed")

    return df


def save_outputs(df):

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # All annotations, including English controls
    df.to_csv(
        ALL_ANNOTATIONS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Primary ML dataset: Telugu positive and negative pairs
    training_data = df[
        df["annotation"].isin(["1", "0"])
    ].copy()

    training_data["label"] = (
        training_data["annotation"].astype(int)
    )

    training_data = training_data.drop(
        columns=["annotation", "pair_id"],
        errors="ignore"
    )

    training_data.to_csv(
        TRAINING_DATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # English controls kept separately
    english_controls = df[
        df["annotation"] == "ENGLISH"
    ].copy()

    english_controls.to_csv(
        ENGLISH_CONTROLS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Ambiguous pairs kept separately
    review_pairs = df[
        df["annotation"] == "REVIEW"
    ].copy()

    review_pairs.to_csv(
        REVIEW_DATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    return training_data, english_controls, review_pairs


def main():

    print("\n" + "=" * 65)
    print("       TELCO-QA | ANNOTATION CONSOLIDATION")
    print("=" * 65)

    # Step 1: Load all four batches
    annotations = load_annotation_batches()

    print("\nTotal annotation records:", len(annotations))

    # Step 2: Validate
    annotations = validate_annotations(annotations)

    # Step 3: Calculate annotation distribution
    counts = annotations["annotation"].value_counts()

    positive = counts.get("1", 0)
    negative = counts.get("0", 0)
    english = counts.get("ENGLISH", 0)
    review = counts.get("REVIEW", 0)

    # Step 4: Save consolidated datasets
    training, controls, review_data = save_outputs(
        annotations
    )

    # Step 5: Display summary
    print("\n" + "-" * 65)
    print("                 ANNOTATION SUMMARY")
    print("-" * 65)

    print(f"Positive Telugu pairs : {positive}")
    print(f"Negative Telugu pairs : {negative}")
    print(f"English controls      : {english}")
    print(f"Review pairs          : {review}")
    print("-" * 65)
    print(f"Total pairs           : {len(annotations)}")

    print("\nML training dataset:")
    print(f"    Total Telugu pairs: {len(training)}")
    print(f"    Positive labels   : {(training['label'] == 1).sum()}")
    print(f"    Negative labels   : {(training['label'] == 0).sum()}")

    print("\nGenerated files:")

    print(f"1. {ALL_ANNOTATIONS_PATH}")
    print(f"2. {TRAINING_DATA_PATH}")
    print(f"3. {ENGLISH_CONTROLS_PATH}")
    print(f"4. {REVIEW_DATA_PATH}")

    print("\n" + "=" * 65)
    print("       CONSOLIDATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
