
"""
Detect potentially incorrect transliteration annotations using Cleanlab.

Uses Logistic Regression out-of-fold probabilities generated
with StratifiedGroupKFold.

Original annotations are never modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cleanlab.filter import find_label_issues


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_oof_predictions_reviewed.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_label_issue_report_reviewed.csv"
)

SELECTED_MODEL = "Logistic Regression"

LABEL_NAMES = {
    0: "Different",
    1: "Same",
}


# --------------------------------------------------
# LOAD AND VALIDATE PREDICTIONS
# --------------------------------------------------

def load_predictions():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "word_1",
        "word_2",
        "label",
        "group_id",
        "model",
        "fold",
        "probability_different",
        "probability_same",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[
        df["model"] == SELECTED_MODEL
    ].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No predictions found for {SELECTED_MODEL}."
        )

    if len(df) < 2:
        raise ValueError(
            "At least two prediction records are required."
        )

    if df["label"].isna().any():
        raise ValueError("Missing labels detected.")

    df["label"] = pd.to_numeric(
        df["label"],
        errors="raise"
    ).astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Labels must be binary: 0 or 1."
        )

    if df["label"].nunique() != 2:
        raise ValueError(
            "Both positive and negative labels must be present."
        )

    probability_columns = [
        "probability_different",
        "probability_same",
    ]

    for column in probability_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    if df[probability_columns].isna().any().any():
        raise ValueError(
            "Missing probabilities detected."
        )

    probabilities = df[
        probability_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite probabilities detected."
        )

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Probability sums are invalid."
        )

    if df["group_id"].isna().any():
        raise ValueError(
            "Missing group IDs detected."
        )

    if df[["word_1", "word_2"]].isna().any().any():
        raise ValueError(
            "Missing word values detected."
        )

    print("\nINPUT VALIDATION")
    print("-" * 45)
    print(f"Selected model : {SELECTED_MODEL}")
    print(f"Total pairs    : {len(df)}")
    print(f"Positive labels: {(df['label'] == 1).sum()}")
    print(f"Negative labels: {(df['label'] == 0).sum()}")

    return df, probabilities


# --------------------------------------------------
# DETECT LABEL ISSUES
# --------------------------------------------------

def detect_issues(df, probabilities):

    labels = df["label"].to_numpy(dtype=int)

    ranked_indices = find_label_issues(
        labels=labels,
        pred_probs=probabilities,
        return_indices_ranked_by="self_confidence",
    )

    ranked_indices = np.asarray(
        ranked_indices,
        dtype=int,
    )

    if (
        ranked_indices.ndim != 1
        or np.any(ranked_indices < 0)
        or np.any(ranked_indices >= len(df))
        or len(np.unique(ranked_indices)) != len(ranked_indices)
    ):
        raise ValueError(
            "Cleanlab returned invalid ranked indices."
        )

    issue_mask = np.zeros(
        len(df),
        dtype=bool,
    )

    issue_mask[ranked_indices] = True

    assigned_label_probability = probabilities[
        np.arange(len(labels)),
        labels,
    ]

    suggested_labels = probabilities.argmax(axis=1)

    report = df.copy()

    report["probability_different"] = probabilities[:, 0]
    report["probability_same"] = probabilities[:, 1]

    report["assigned_label_probability"] = (
        assigned_label_probability
    )

    report["suggested_label"] = suggested_labels

    report["suggested_label_name"] = [
        LABEL_NAMES[int(label)]
        for label in suggested_labels
    ]

    report["cleanlab_flagged"] = issue_mask

    rank_lookup = {
        int(index): rank
        for rank, index in enumerate(
            ranked_indices,
            start=1,
        )
    }

    report["cleanlab_rank"] = [
        rank_lookup.get(index, np.nan)
        for index in range(len(report))
    ]

    report["model_disagrees"] = (
        report["label"] != report["suggested_label"]
    )

    report["label_issue_status"] = np.where(
        report["cleanlab_flagged"],
        "REVIEW",
        "NO_FLAG",
    )

    return report


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

def save_report(report):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = report.sort_values(
        by=[
            "cleanlab_flagged",
            "cleanlab_rank",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    flagged = report[
        report["cleanlab_flagged"]
    ]

    print("\n" + "=" * 60)
    print("CONFIDENT LEARNING RESULTS")
    print("=" * 60)

    print(f"Total pairs examined : {len(report)}")
    print(f"Potential label issues: {len(flagged)}")
    print(
        f"Unflagged pairs       : "
        f"{len(report) - len(flagged)}"
    )

    if not flagged.empty:

        print("\nSUSPECTED LABEL ISSUES")
        print("-" * 60)

        display_columns = [
            "word_1",
            "word_2",
            "label",
            "suggested_label",
            "assigned_label_probability",
            "cleanlab_rank",
            "group_id",
        ]

        print(
            flagged[display_columns].to_string(
                index=False
            )
        )

    else:
        print("\nCleanlab did not flag any pairs.")

    print("\nREPORT SAVED")
    print(OUTPUT_FILE)

    print("\nOriginal annotations were not modified.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nTRANSLITERATION LABEL ERROR DETECTION")
    print("=" * 60)

    df, probabilities = load_predictions()

    report = detect_issues(
        df,
        probabilities,
    )

    save_report(report)

    print("\nDetection completed successfully.")


if __name__ == "__main__":
    main()
