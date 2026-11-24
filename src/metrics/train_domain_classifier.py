"""
Metric 05 — Train a Telugu-English Domain Classifier

Learns domain labels from the dataset's existing `topic` column.
This script trains and evaluates a TF-IDF + Logistic Regression model.

Run from the TelCo-QA repository root:
    python src/metrics/train_domain_classifier.py

Expected dataset:
    data/raw/TelCo-QA-Dataset.csv.xlsx
or:
    data/raw/TelCo-QA-Dataset.xlsx
or a CSV file with the same columns.

Outputs:
    outputs/metric_05_domain_classifier/
        domain_classifier.joblib
        evaluation_report.json
        test_predictions.csv
        confusion_matrix.csv
"""

from pathlib import Path
import json
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_CANDIDATES = [
    ROOT_DIR / "data" / "raw" / "TelCo-QA-Dataset.csv.xlsx",
    ROOT_DIR / "data" / "raw" / "TelCo-QA-Dataset.xlsx",
    ROOT_DIR / "data" / "raw" / "TelCo-QA-Dataset.csv",
    ROOT_DIR / "TelCo-QA-Dataset.xlsx",
]

TEXT_COLUMN = "text"
LABEL_COLUMN = "topic"
TEST_SIZE = 0.20
RANDOM_STATE = 42

OUTPUT_DIR = ROOT_DIR / "outputs" / "metric_05_domain_classifier"


def find_dataset() -> Path:
    """Find the dataset from the configured candidate paths."""
    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(f"  - {path}" for path in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Dataset file was not found. Checked these locations:\n"
        f"{searched}\n\n"
        "Update DATA_CANDIDATES in this script to match your file location."
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV or Excel data based on the file extension."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def normalize_for_grouping(text: str) -> str:
    """
    Normalize text only for grouping duplicates/near-identical whitespace
    and punctuation variants during the train/test split.
    The original text is still used for model training.
    """
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and remove rows without usable text or labels."""
    missing = [col for col in [TEXT_COLUMN, LABEL_COLUMN] if col not in df.columns]
    if missing:
        raise KeyError(
            f"Required column(s) missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    data = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna("").astype(str).str.strip()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype("string").str.strip()

    # Remove empty text and missing/empty topic labels.
    data = data[
        (data[TEXT_COLUMN] != "")
        & data[LABEL_COLUMN].notna()
        & (data[LABEL_COLUMN] != "")
    ].copy()

    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str)
    data["group_key"] = data[TEXT_COLUMN].map(normalize_for_grouping)

    # A blank group key should not occur after filtering, but guard anyway.
    data = data[data["group_key"] != ""].reset_index(drop=True)

    if data[LABEL_COLUMN].nunique() < 2:
        raise ValueError("At least two distinct topic labels are required.")

    return data


def split_without_duplicate_leakage(data: pd.DataFrame):
    """
    Keep identical normalized text in only one split, reducing duplicate
    leakage between training and test data.

    GroupShuffleSplit is used because repeated texts can occur under multiple
    rows. The split is group-aware, but not perfectly stratified.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            data[TEXT_COLUMN],
            data[LABEL_COLUMN],
            groups=data["group_key"],
        )
    )

    train_data = data.iloc[train_indices].copy()
    test_data = data.iloc[test_indices].copy()

    train_labels = set(train_data[LABEL_COLUMN])
    unseen_test_labels = sorted(set(test_data[LABEL_COLUMN]) - train_labels)

    if unseen_test_labels:
        print(
            "\nWARNING: These label(s) appear in the test split but not in "
            f"training: {unseen_test_labels}"
        )
        print(
            "Their test predictions cannot be learned by this classifier. "
            "Check whether the dataset has very few examples for these labels."
        )

    return train_data, test_data


def build_model() -> Pipeline:
    """
    Word n-grams capture domain terms and phrases.
    Character n-grams help with romanized Telugu spelling variation.
    """
    features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100000,
                    sublinear_tf=True,
                    strip_accents=None,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=150000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("features", features),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    dataset_path = find_dataset()
    print(f"Dataset: {dataset_path}")

    raw_data = load_dataset(dataset_path)
    data = prepare_data(raw_data)

    print(f"Usable records: {len(data):,}")
    print(f"Domain classes: {data[LABEL_COLUMN].nunique()}")
    print("\nLabel distribution:")
    print(data[LABEL_COLUMN].value_counts().to_string())

    train_data, test_data = split_without_duplicate_leakage(data)

    print(f"\nTraining records: {len(train_data):,}")
    print(f"Testing records:  {len(test_data):,}")

    model = build_model()

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(train_data[TEXT_COLUMN], train_data[LABEL_COLUMN])

    predictions = model.predict(test_data[TEXT_COLUMN])

    accuracy = accuracy_score(test_data[LABEL_COLUMN], predictions)
    report_dict = classification_report(
        test_data[LABEL_COLUMN],
        predictions,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        test_data[LABEL_COLUMN],
        predictions,
        zero_division=0,
    )

    labels = sorted(data[LABEL_COLUMN].unique())
    matrix = confusion_matrix(
        test_data[LABEL_COLUMN],
        predictions,
        labels=labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "domain_classifier.joblib"
    report_path = OUTPUT_DIR / "evaluation_report.json"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    matrix_path = OUTPUT_DIR / "confusion_matrix.csv"

    joblib.dump(model, model_path)

    report_payload = {
        "dataset_path": str(dataset_path),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "model": "TF-IDF (word + character n-grams) + Logistic Regression",
        "split_method": "GroupShuffleSplit on normalized text",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "total_usable_records": int(len(data)),
        "training_records": int(len(train_data)),
        "testing_records": int(len(test_data)),
        "classes": labels,
        "accuracy": float(accuracy),
        "classification_report": report_dict,
        "note": (
            "Evaluation uses existing topic labels as targets. "
            "Metrics measure agreement with those labels, not independent "
            "human-verified domain correctness."
        ),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report_payload, file, indent=4, ensure_ascii=False)

    prediction_output = test_data[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    prediction_output = prediction_output.rename(
        columns={LABEL_COLUMN: "actual_topic"}
    )
    prediction_output["predicted_topic"] = predictions
    prediction_output.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    matrix_df.to_csv(matrix_path, encoding="utf-8-sig")

    print("\nTEST RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report_text)

    print("\nSaved files:")
    print(f"  Model:             {model_path}")
    print(f"  Evaluation report: {report_path}")
    print(f"  Test predictions:  {predictions_path}")
    print(f"  Confusion matrix:  {matrix_path}")


if __name__ == "__main__":
    main()
