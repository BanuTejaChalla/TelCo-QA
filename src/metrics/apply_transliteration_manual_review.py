"""
Apply user-confirmed manual review decisions to transliteration annotations.

This script NEVER overwrites the original annotation batches or the original
consolidated annotation/training files. It creates separate reviewed outputs.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_ANNOTATIONS = DATA_DIR / "transliteration_annotations_all.csv"
DECISIONS_FILE = DATA_DIR / "transliteration_manual_review_decisions.csv"

OUTPUT_ANNOTATIONS = DATA_DIR / "transliteration_annotations_reviewed.csv"
OUTPUT_TRAINING = DATA_DIR / "transliteration_ml_training_data_reviewed.csv"
OUTPUT_AUDIT = DATA_DIR / "transliteration_manual_review_audit.csv"


def normalize_word(value):
    return str(value).strip().lower()


def normalize_pair(word_1, word_2):
    return tuple(sorted((normalize_word(word_1), normalize_word(word_2))))


def load_inputs():
    if not INPUT_ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"Original consolidated annotations not found: {INPUT_ANNOTATIONS}\n"
            "Run the existing annotation consolidation step first."
        )

    if not DECISIONS_FILE.exists():
        raise FileNotFoundError(
            f"Manual review decisions file not found: {DECISIONS_FILE}"
        )

    annotations = pd.read_csv(
        INPUT_ANNOTATIONS,
        dtype=str,
        keep_default_na=False,
    )
    decisions = pd.read_csv(
        DECISIONS_FILE,
        dtype=str,
        keep_default_na=False,
    )

    required_annotation_columns = {"word_1", "word_2", "annotation"}
    required_decision_columns = {
        "word_1", "word_2", "original_label",
        "reviewed_label", "review_status", "review_notes"
    }

    missing_annotations = required_annotation_columns - set(annotations.columns)
    missing_decisions = required_decision_columns - set(decisions.columns)

    if missing_annotations:
        raise ValueError(
            f"Original annotations missing columns: {sorted(missing_annotations)}"
        )
    if missing_decisions:
        raise ValueError(
            f"Decision file missing columns: {sorted(missing_decisions)}"
        )

    return annotations, decisions


def apply_decisions(annotations, decisions):
    reviewed = annotations.copy()
    reviewed["pair_key"] = reviewed.apply(
        lambda row: normalize_pair(row["word_1"], row["word_2"]),
        axis=1,
    )

    audit_rows = []
    seen_pairs = set()

    for _, decision in decisions.iterrows():
        pair = normalize_pair(decision["word_1"], decision["word_2"])

        if pair in seen_pairs:
            raise ValueError(f"Duplicate decision found for pair: {pair}")
        seen_pairs.add(pair)

        if decision["review_status"].strip().upper() != "CONFIRMED":
            raise ValueError(f"Decision is not confirmed for pair: {pair}")

        matches = reviewed.index[reviewed["pair_key"] == pair].tolist()

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one annotation for {pair}; found {len(matches)}."
            )

        row_index = matches[0]
        existing_label = str(reviewed.at[row_index, "annotation"]).strip()
        expected_original = str(decision["original_label"]).strip()
        new_label = str(decision["reviewed_label"]).strip()

        if existing_label not in {"0", "1"}:
            raise ValueError(
                f"Expected a binary original annotation for {pair}; "
                f"found {existing_label!r}."
            )

        if existing_label != expected_original:
            raise ValueError(
                f"Original label mismatch for {pair}: expected "
                f"{expected_original}, found {existing_label}. "
                "No reviewed outputs have been written."
            )

        if new_label not in {"0", "1"}:
            raise ValueError(f"Invalid reviewed label {new_label!r} for {pair}")

        reviewed.at[row_index, "annotation"] = new_label

        audit_rows.append({
            "word_1": decision["word_1"],
            "word_2": decision["word_2"],
            "original_label": existing_label,
            "reviewed_label": new_label,
            "changed": existing_label != new_label,
            "review_status": decision["review_status"],
            "review_notes": decision["review_notes"],
        })

    reviewed = reviewed.drop(columns=["pair_key"])

    audit = pd.DataFrame(audit_rows)
    return reviewed, audit


def save_reviewed_outputs(reviewed, audit):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    reviewed.to_csv(
        OUTPUT_ANNOTATIONS,
        index=False,
        encoding="utf-8-sig",
    )

    training = reviewed[
        reviewed["annotation"].isin(["0", "1"])
    ].copy()

    training["label"] = training["annotation"].astype(int)
    training = training.drop(
        columns=["annotation", "pair_id"],
        errors="ignore",
    )

    training.to_csv(
        OUTPUT_TRAINING,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    return training


def main():
    print("\n" + "=" * 68)
    print("       TELCO-QA | APPLY MANUAL TRANSLITERATION REVIEW")
    print("=" * 68)

    annotations, decisions = load_inputs()
    reviewed, audit = apply_decisions(annotations, decisions)
    training = save_reviewed_outputs(reviewed, audit)

    original_counts = annotations["annotation"].value_counts()
    reviewed_counts = reviewed["annotation"].value_counts()

    print(f"\nOriginal annotation records : {len(annotations)}")
    print(f"Manual decisions applied   : {len(audit)}")
    print(f"Labels actually changed    : {int(audit['changed'].sum())}")

    print("\nLABEL COUNTS")
    print("-" * 45)
    print(
        f"Original positives : {original_counts.get('1', 0)}"
        f" -> Reviewed: {reviewed_counts.get('1', 0)}"
    )
    print(
        f"Original negatives : {original_counts.get('0', 0)}"
        f" -> Reviewed: {reviewed_counts.get('0', 0)}"
    )
    print(
        f"English controls   : {reviewed_counts.get('ENGLISH', 0)}"
    )
    print(
        f"Review labels      : {reviewed_counts.get('REVIEW', 0)}"
    )

    print("\nREVIEWED ML TRAINING DATA")
    print("-" * 45)
    print(f"Total Telugu pairs : {len(training)}")
    print(f"Positive labels    : {(training['label'] == 1).sum()}")
    print(f"Negative labels    : {(training['label'] == 0).sum()}")

    print("\nGENERATED FILES")
    print("-" * 45)
    print(OUTPUT_ANNOTATIONS)
    print(OUTPUT_TRAINING)
    print(OUTPUT_AUDIT)

    print("\nOriginal annotation files were not modified.")
    print("Manual review application completed successfully.")


if __name__ == "__main__":
    main()
