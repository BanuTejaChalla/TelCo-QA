
"""
Train and compare transliteration classifiers.

Uses:
- 18 explicitly selected numerical features
- 3-fold StratifiedGroupKFold
- Group-aware cross-validation
- Out-of-fold predictions
- Out-of-fold class probabilities for Cleanlab

Input:
    data/processed/transliteration_grouped_dataset.csv

Outputs:
    data/processed/transliteration_model_comparison.csv
    data/processed/transliteration_oof_predictions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_grouped_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

N_SPLITS = 3
RANDOM_STATE = 42


# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

FEATURE_COLUMNS = [
    "length_1",
    "length_2",
    "length_difference",
    "length_ratio",
    "edit_distance",
    "normalized_edit_distance",
    "similarity_ratio",
    "common_prefix_length",
    "common_suffix_length",
    "vowel_count_1",
    "vowel_count_2",
    "vowel_count_difference",
    "consonant_count_1",
    "consonant_count_2",
    "consonant_count_difference",
    "repeated_vowels_1",
    "repeated_vowels_2",
    "repeated_vowel_difference",
]


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def get_models():

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    return models


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = (
        FEATURE_COLUMNS
        + ["label", "group_id", "word_1", "word_2"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["label"].isna().any():
        raise ValueError("Missing target labels detected.")

    if df["group_id"].isna().any():
        raise ValueError("Missing group IDs detected.")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)
    groups = df["group_id"]

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    if X.isna().any().any():
        raise ValueError("Missing feature values detected.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values detected.")

    if y.nunique() != 2:
        raise ValueError("Both classes must be present.")

    print("\nDATASET SUMMARY")
    print("-" * 45)
    print(f"Total pairs       : {len(df)}")
    print(f"Total features    : {len(FEATURE_COLUMNS)}")
    print(f"Unique groups     : {groups.nunique()}")
    print(f"Positive labels   : {(y == 1).sum()}")
    print(f"Negative labels   : {(y == 0).sum()}")

    return df, X, y, groups


# --------------------------------------------------
# CROSS-VALIDATION AND PROBABILITY GENERATION
# --------------------------------------------------

def evaluate_models(df, X, y, groups):

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = get_models()

    comparison_results = []
    all_predictions = []

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"MODEL: {model_name}")
        print("=" * 60)

        # Out-of-fold predictions
        oof_predictions = np.full(
            len(df), -1, dtype=int
        )

        # Probability columns:
        # Column 0 = P(different)
        # Column 1 = P(same)
        oof_probabilities = np.full(
            (len(df), 2),
            np.nan,
            dtype=float,
        )

        fold_numbers = np.full(
            len(df), -1, dtype=int
        )

        for fold_number, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])

            overlap = train_groups.intersection(test_groups)

            if overlap:
                raise RuntimeError(
                    f"Group leakage detected in fold {fold_number}."
                )

            fold_model = clone(model)

            fold_model.fit(X_train, y_train)

            # Hard predictions
            predictions = fold_model.predict(X_test)

            # Probability predictions
            probabilities = fold_model.predict_proba(X_test)

            # Explicitly map probability columns to class labels.
            class_positions = {
                int(label): position
                for position, label
                in enumerate(fold_model.classes_)
            }

            if set(class_positions) != {0, 1}:
                raise RuntimeError(
                    f"Unexpected class ordering in fold {fold_number}."
                )

            probability_different = probabilities[
                :, class_positions[0]
            ]

            probability_same = probabilities[
                :, class_positions[1]
            ]

            # Store predictions at their original row positions.
            oof_predictions[test_idx] = predictions

            oof_probabilities[test_idx, 0] = (
                probability_different
            )

            oof_probabilities[test_idx, 1] = (
                probability_same
            )

            fold_numbers[test_idx] = fold_number

            fold_f1 = f1_score(
                y_test,
                predictions,
                zero_division=0,
            )

            print(
                f"Fold {fold_number}: "
                f"Train={len(train_idx)}, "
                f"Test={len(test_idx)}, "
                f"F1={fold_f1:.4f}, "
                f"Group overlap={len(overlap)}"
            )

        # Validate complete OOF results.
        if (oof_predictions == -1).any():
            raise RuntimeError(
                f"Missing predictions for {model_name}."
            )

        if np.isnan(oof_probabilities).any():
            raise RuntimeError(
                f"Missing probabilities for {model_name}."
            )

        if not np.allclose(
            oof_probabilities.sum(axis=1),
            1.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Invalid probability sums for {model_name}."
            )

        # --------------------------------------------------
        # EVALUATION METRICS
        # --------------------------------------------------

        accuracy = accuracy_score(y, oof_predictions)

        balanced_accuracy = balanced_accuracy_score(
            y, oof_predictions
        )

        precision = precision_score(
            y, oof_predictions, zero_division=0
        )

        recall = recall_score(
            y, oof_predictions, zero_division=0
        )

        f1 = f1_score(
            y, oof_predictions, zero_division=0
        )

        macro_f1 = f1_score(
            y,
            oof_predictions,
            average="macro",
            zero_division=0,
        )

        matrix = confusion_matrix(
            y,
            oof_predictions,
            labels=[0, 1],
        )

        print("\nOUT-OF-FOLD RESULTS")
        print("-" * 45)
        print(f"Accuracy          : {accuracy:.4f}")
        print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1-score          : {f1:.4f}")
        print(f"Macro F1          : {macro_f1:.4f}")

        print("\nConfusion Matrix")
        print("Rows = Actual, Columns = Predicted")
        print(matrix)

        print("\nClassification Report")

        print(
            classification_report(
                y,
                oof_predictions,
                labels=[0, 1],
                target_names=[
                    "Different (0)",
                    "Same (1)",
                ],
                zero_division=0,
            )
        )

        # --------------------------------------------------
        # SAVE COMPARISON RESULTS
        # --------------------------------------------------

        comparison_results.append({
            "model": model_name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "macro_f1": macro_f1,
        })

        # --------------------------------------------------
        # SAVE OOF PREDICTIONS AND PROBABILITIES
        # --------------------------------------------------

        model_predictions = df[
            ["word_1", "word_2", "label", "group_id"]
        ].copy()

        model_predictions["model"] = model_name

        model_predictions["fold"] = fold_numbers

        model_predictions["predicted_label"] = (
            oof_predictions
        )

        model_predictions["probability_different"] = (
            oof_probabilities[:, 0]
        )

        model_predictions["probability_same"] = (
            oof_probabilities[:, 1]
        )

        model_predictions["correct"] = (
            model_predictions["label"]
            == model_predictions["predicted_label"]
        )

        all_predictions.append(model_predictions)

        print(
            "\nOut-of-fold probabilities generated:",
            len(model_predictions),
        )

    return comparison_results, all_predictions


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(comparison_results, all_predictions):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df = comparison_df.sort_values(
        by="macro_f1",
        ascending=False,
    )

    comparison_path = (
        OUTPUT_DIR / "transliteration_model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = (
        OUTPUT_DIR / "transliteration_oof_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(comparison_df.to_string(index=False))

    print("\nFILES SAVED")
    print("-" * 45)
    print(comparison_path)
    print(predictions_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nTRANSLITERATION MODEL EVALUATION")
    print("=" * 60)

    df, X, y, groups = load_dataset()

    comparison_results, all_predictions = evaluate_models(
        df,
        X,
        y,
        groups,
    )

    save_results(
        comparison_results,
        all_predictions,
    )

    print("\nEvaluation and probability generation completed.")


if __name__ == "__main__":
    main()
