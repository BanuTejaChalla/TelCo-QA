
"""
Metric 04 - Annotation Availability
===================================

Purpose:
    Measure the availability and completeness of annotation fields
    in a dataset.

This metric does NOT evaluate:
    - Annotation correctness
    - Annotation agreement
    - Annotation reliability
    - Semantic quality of labels

Scoring:
    Column Availability =
        (Non-empty values / Total records) * 100

    Overall Score =
        Mean availability across detected annotation columns

Quality Tiers:
    High   : 90% - 100%
    Medium : 60% - <90%
    Low    : <60%

If no annotation columns are detected, the metric is N/A.
If the dataset contains zero records, the metric is N/A.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

METRIC_NAME = "Metric 04 - Annotation Availability"

HIGH_THRESHOLD = 90.0
MEDIUM_THRESHOLD = 60.0

DEFAULT_OUTPUT_DIR = Path(
    "outputs/metric_04_annotation_availability"
)

# Column names commonly used for annotations.
ANNOTATION_NAME_PATTERNS = [
    r"(^|_)(label|labels)(_|$)",
    r"(^|_)(class|classes)(_|$)",
    r"(^|_)(target|targets)(_|$)",
    r"(^|_)(sentiment)(_|$)",
    r"(^|_)(emotion|emotions)(_|$)",
    r"(^|_)(topic|topics)(_|$)",
    r"(^|_)(category|categories)(_|$)",
    r"(^|_)(annotation|annotations)(_|$)",
    r"(^|_)(rating|ratings)(_|$)",
    r"(^|_)(intent|intents)(_|$)",
    r"(^|_)(outcome)(_|$)",
    r"(^|_)(ground_truth|gold_label)(_|$)",
]

# Metadata fields that should not be treated as actual annotations.
EXCLUDED_NAME_PATTERNS = [
    r"(^|_)(annotation_method|labeling_method)(_|$)",
    r"(^|_)(annotator|annotators)(_|$)",
    r"(^|_)(annotation_source|label_source)(_|$)",
    r"(^|_)(annotation_status|label_status)(_|$)",
    r"(^|_)(annotation_confidence|label_confidence)(_|$)",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(column_name):
    """
    Convert a column name into a normalized form.

    Example:
        'Gold Label' -> 'gold_label'
        'Sentiment-Label' -> 'sentiment_label'
    """

    name = str(column_name).strip().lower()

    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)

    return name.strip("_")


def is_annotation_column(column_name):
    """
    Determine whether a column represents annotation information.

    Detection is based on common annotation-related column names.
    """

    normalized_name = normalize_column_name(column_name)

    # Exclude annotation metadata that does not contain the label itself.
    for pattern in EXCLUDED_NAME_PATTERNS:
        if re.search(pattern, normalized_name):
            return False

    # Identify annotation columns.
    for pattern in ANNOTATION_NAME_PATTERNS:
        if re.search(pattern, normalized_name):
            return True

    return False


def is_missing_annotation(value):
    """
    Return True when a value represents missing annotation.

    Empty strings and common missing-value placeholders are treated
    as missing.

    Numeric zero and Boolean False are valid annotation values.
    """

    if value is None:
        return True

    # Handle lists, tuples, sets, and dictionaries.
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    # Handle pandas and NumPy missing values.
    try:
        missing_result = pd.isna(value)

        if isinstance(missing_result, bool):
            if missing_result:
                return True

        elif hasattr(missing_result, "item"):
            if bool(missing_result.item()):
                return True

    except (TypeError, ValueError):
        pass

    # Handle string values.
    if isinstance(value, str):
        normalized_value = value.strip().lower()

        missing_placeholders = {
            "",
            "nan",
            "none",
            "null",
            "na",
            "n/a",
            "not available",
            "not annotated",
            "unannotated",
            "unlabelled",
            "unlabeled",
        }

        if normalized_value in missing_placeholders:
            return True

    return False


def get_quality_tier(score):
    """
    Assign a quality tier based on the overall availability score.
    """

    if score >= HIGH_THRESHOLD:
        return "High"

    if score >= MEDIUM_THRESHOLD:
        return "Medium"

    return "Low"


# ============================================================
# CORE METRIC
# ============================================================

def calculate_annotation_availability(
    df,
    output_dir=None,
    annotation_columns=None,
):
    """
    Calculate Metric 04 - Annotation Availability.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to evaluate.

    output_dir : str or pathlib.Path, optional
        Directory where reports will be saved.

    annotation_columns : list, optional
        Explicit annotation columns to evaluate.

        If None, annotation columns are detected automatically.

    Returns
    -------
    dict
        Metric summary, column report, and record report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    # Preserve the original dataset.
    working_df = df.copy()

    total_records = len(working_df)

    # --------------------------------------------------------
    # 1. Detect annotation columns
    # --------------------------------------------------------

    if annotation_columns is not None:

        missing_columns = [
            column
            for column in annotation_columns
            if column not in working_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "The following annotation columns do not exist: "
                f"{missing_columns}"
            )

        detected_columns = list(annotation_columns)

    else:

        detected_columns = [
            column
            for column in working_df.columns
            if is_annotation_column(column)
        ]

    # --------------------------------------------------------
    # 2. Handle N/A conditions
    # --------------------------------------------------------

    if total_records == 0:

        summary = {
            "metric": METRIC_NAME,
            "status": "N/A",
            "reason": "Dataset contains zero records.",
            "total_records": 0,
            "annotation_columns": detected_columns,
            "overall_score": None,
            "quality_tier": "N/A",
        }

        return {
            "summary": summary,
            "column_report": pd.DataFrame(),
            "record_report": working_df,
        }

    if len(detected_columns) == 0:

        summary = {
            "metric": METRIC_NAME,
            "status": "N/A",
            "reason": (
                "No annotation columns were detected. "
                "Provide annotation_columns explicitly if needed."
            ),
            "total_records": total_records,
            "annotation_columns": [],
            "overall_score": None,
            "quality_tier": "N/A",
        }

        return {
            "summary": summary,
            "column_report": pd.DataFrame(),
            "record_report": working_df,
        }

    # --------------------------------------------------------
    # 3. Calculate availability for each annotation column
    # --------------------------------------------------------

    column_results = {}

    for column in detected_columns:

        missing_mask = working_df[column].apply(
            is_missing_annotation
        )

        available_count = int((~missing_mask).sum())
        missing_count = int(missing_mask.sum())

        availability_percentage = (
            available_count / total_records
        ) * 100

        column_results[column] = {
            "column_name": str(column),
            "total_records": total_records,
            "available_count": available_count,
            "missing_count": missing_count,
            "availability_percentage": round(
                availability_percentage, 2
            ),
        }

    column_report = pd.DataFrame(
        list(column_results.values())
    )

    # --------------------------------------------------------
    # 4. Calculate record-level completeness
    # --------------------------------------------------------

    annotation_missing_matrix = pd.DataFrame(
        {
            column: working_df[column].apply(
                is_missing_annotation
            )
            for column in detected_columns
        },
        index=working_df.index,
    )

    annotation_available_matrix = (
        ~annotation_missing_matrix
    )

    available_annotation_count = (
        annotation_available_matrix.sum(axis=1)
    )

    missing_annotation_count = (
        annotation_missing_matrix.sum(axis=1)
    )

    has_any_annotation = (
        available_annotation_count > 0
    )

    has_all_annotations = (
        available_annotation_count
        == len(detected_columns)
    )

    # --------------------------------------------------------
    # 5. Calculate overall score
    # --------------------------------------------------------

    component_scores = [
        result["availability_percentage"]
        for result in column_results.values()
    ]

    overall_score = sum(component_scores) / len(
        component_scores
    )

    overall_score = round(overall_score, 2)

    quality_tier = get_quality_tier(
        overall_score
    )

    # --------------------------------------------------------
    # 6. Build record-level report
    # --------------------------------------------------------

    record_report = working_df.copy()

    record_report[
        "metric04_available_annotation_count"
    ] = available_annotation_count

    record_report[
        "metric04_missing_annotation_count"
    ] = missing_annotation_count

    record_report[
        "metric04_has_any_annotation"
    ] = has_any_annotation

    record_report[
        "metric04_has_all_annotations"
    ] = has_all_annotations

    # --------------------------------------------------------
    # 7. Build summary
    # --------------------------------------------------------

    records_with_any_annotation = int(
        has_any_annotation.sum()
    )

    records_with_all_annotations = int(
        has_all_annotations.sum()
    )

    summary = {
        "metric": METRIC_NAME,
        "status": "Evaluated",
        "total_records": total_records,
        "annotation_column_count": len(
            detected_columns
        ),
        "annotation_columns": [
            str(column)
            for column in detected_columns
        ],
        "overall_score": overall_score,
        "quality_tier": quality_tier,
        "records_with_any_annotation": (
            records_with_any_annotation
        ),
        "records_with_all_annotations": (
            records_with_all_annotations
        ),
        "records_without_any_annotation": (
            total_records
            - records_with_any_annotation
        ),
        "column_availability": {
            str(column): result[
                "availability_percentage"
            ]
            for column, result in column_results.items()
        },
        "scoring_method": (
            "Equal-weighted mean of annotation "
            "column availability percentages."
        ),
        "thresholds": {
            "high": HIGH_THRESHOLD,
            "medium": MEDIUM_THRESHOLD,
        },
    }

    # --------------------------------------------------------
    # 8. Save reports
    # --------------------------------------------------------

    if output_dir is not None:

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        column_report_path = (
            output_dir
            / "metric_04_annotation_availability_columns.csv"
        )

        record_report_path = (
            output_dir
            / "metric_04_annotation_availability_records.csv"
        )

        summary_path = (
            output_dir
            / "metric_04_annotation_availability_summary.json"
        )

        column_report.to_csv(
            column_report_path,
            index=False,
            encoding="utf-8-sig",
        )

        record_report.to_csv(
            record_report_path,
            index=False,
            encoding="utf-8-sig",
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False,
            )

        summary["output_files"] = {
            "column_report": str(
                column_report_path
            ),
            "record_report": str(
                record_report_path
            ),
            "summary": str(
                summary_path
            ),
        }

    return {
        "summary": summary,
        "column_report": column_report,
        "record_report": record_report,
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_metric_04_results(result):
    """
    Print a readable Metric 04 summary.
    """

    summary = result["summary"]

    print("\n" + "=" * 60)
    print("METRIC 04 - ANNOTATION AVAILABILITY")
    print("=" * 60)

    print(f"Status: {summary['status']}")

    if summary["status"] == "N/A":

        print(f"Reason: {summary['reason']}")
        print("=" * 60)

        return

    print(
        f"Total records: "
        f"{summary['total_records']}"
    )

    print(
        f"Annotation columns detected: "
        f"{summary['annotation_column_count']}"
    )

    print(
        f"Overall availability score: "
        f"{summary['overall_score']}%"
    )

    print(
        f"Overall quality tier: "
        f"{summary['quality_tier']}"
    )

    print("\nANNOTATION COLUMN AVAILABILITY")

    print(
        result["column_report"].to_string(
            index=False
        )
    )

    print("\nRECORD-LEVEL COMPLETENESS")

    print(
        f"Records with any annotation: "
        f"{summary['records_with_any_annotation']}"
    )

    print(
        f"Records with all annotations: "
        f"{summary['records_with_all_annotations']}"
    )

    print(
        f"Records without annotations: "
        f"{summary['records_without_any_annotation']}"
    )

    print("=" * 60)


# ============================================================
# STANDALONE EXECUTION
# ============================================================

def load_dataset(file_path):
    """
    Load a CSV or Excel dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    if extension in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError(
        "Unsupported file format. "
        "Use CSV, XLSX, or XLS."
    )


def main():
    """
    Standalone execution entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Metric 04 - Annotation Availability"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the dataset CSV or Excel file.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for Metric 04 output reports.",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.input)

    result = calculate_annotation_availability(
        dataset,
        output_dir=args.output_dir,
    )

    print_metric_04_results(result)


if __name__ == "__main__":
    main()
