
"""
Metric 04 - Annotation Reliability
Yama-assisted, abstention-aware annotation verification.

Workflow:
1. Detect valid annotation columns.
2. If no valid annotations exist, assign score 0 and exclude this metric
   from the overall score.
3. Use Yama to independently predict labels from the text.
4. Compare predictions against the original dataset annotations.
5. Report prediction coverage, agreement, and abstentions separately.
6. Preserve all original annotations.

Important:
Yama agreement is a model-based reliability estimate, not human ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"
)

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "metric_04"

MODEL_ID = "LatentAILabs/yama-indic"

DEFAULT_BATCH_SIZE = 16


ANNOTATION_PATTERNS = (
    r"labels?",
    r"annotations?",
    r"targets?",
    r"classes?",
    r"category",
    r"sentiment",
    r"intents?",
    r"tags?",
    r"ner",
    r"entity_?labels?",
    r"topics?",
    r"rating",
    r"answers?",
    r"ground_truth",
    r"gold_?labels?",
)

EXCLUDED_PATTERNS = (
    r"id",
    r".*_id",
    r"text",
    r"sentence",
    r"utterance",
    r"content",
    r"review",
    r"comment",
    r"source",
    r"dataset",
    r"split",
    r"partition",
    r"script",
    r"language",
    r"annotation_method",
    r"annotator(_id)?",
    r"created_at",
    r"updated_at",
    r"timestamp",
    r"url",
)

TEXT_COLUMN_CANDIDATES = (
    "text",
    "sentence",
    "utterance",
    "message",
    "content",
    "review",
    "comment",
)

MISSING_LABELS = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def normalize_column(name: Any) -> str:
    """Normalize a column name for matching."""
    value = str(name).strip().lower()
    value = re.sub(r"[\-/]+", "_", value)
    return re.sub(r"\s+", "_", value)


def normalized_value(value: Any) -> str:
    """Convert a value to a clean string, treating NaN as empty."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def valid_value(value: Any) -> bool:
    """Return whether a value is a usable annotation or text."""
    return normalized_value(value).casefold() not in MISSING_LABELS


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

def detect_annotation_columns(
    df: pd.DataFrame,
    explicit: list[str] | None = None,
) -> list[str]:
    """
    Detect annotation columns.

    Explicitly supplied columns take precedence.
    """
    if explicit:
        missing = [
            column
            for column in explicit
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Annotation columns not found: {missing}"
            )

        return explicit

    detected = []

    for column in df.columns:
        name = normalize_column(column)

        if any(
            re.fullmatch(pattern, name)
            for pattern in EXCLUDED_PATTERNS
        ):
            continue

        if any(
            re.fullmatch(pattern, name)
            for pattern in ANNOTATION_PATTERNS
        ):
            detected.append(column)

    return detected


def detect_text_column(
    df: pd.DataFrame,
    explicit: str | None = None,
) -> str:
    """Identify the text column."""
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Text column '{explicit}' not found."
            )

        return explicit

    normalized_columns = {
        normalize_column(column): column
        for column in df.columns
    }

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    raise ValueError(
        "Could not identify the text column. "
        "Specify it using --text-column."
    )


# ---------------------------------------------------------------------
# Label descriptions and Yama prompt
# ---------------------------------------------------------------------

def label_description(
    column: str,
    label: str,
) -> str:
    """
    Provide Yama with a useful explanation of each label.

    Descriptions are intentionally generic where the label meaning
    cannot safely be inferred from its name.
    """
    col = normalize_column(column)
    key = label.strip().casefold()

    if "sentiment" in col or "emotion" in col:
        descriptions = {
            "positive": (
                "Positive sentiment: expresses approval, satisfaction, "
                "praise, happiness, or a favorable opinion."
            ),
            "pos": (
                "Positive sentiment: expresses approval, satisfaction, "
                "praise, happiness, or a favorable opinion."
            ),
            "negative": (
                "Negative sentiment: expresses criticism, dissatisfaction, "
                "dislike, sadness, or an unfavorable opinion."
            ),
            "neg": (
                "Negative sentiment: expresses criticism, dissatisfaction, "
                "dislike, sadness, or an unfavorable opinion."
            ),
            "neutral": (
                "Neutral sentiment: expresses information or a statement "
                "without clear positive or negative sentiment."
            ),
            "mixed": (
                "Mixed sentiment: contains both positive and negative "
                "sentiment."
            ),
        }

        if key in descriptions:
            return descriptions[key]

        return (
            f"Sentiment or emotion category: '{label}'. "
            "Interpret the label using the sentence's meaning."
        )

    if (
        "topic" in col
        or "domain" in col
        or "category" in col
    ):
        return (
            f"Topic/category label '{label}': "
            "the main subject or theme of the text."
        )

    return (
        f"The dataset's annotation value '{label}' "
        f"for the '{column}' field."
    )


def question_for_column(
    column: str,
    labels: list[str],
) -> dict:
    """
    Construct the Yama classification prompt.

    The prompt encourages classification of short, colloquial,
    code-mixed and Romanized Telugu text without forcing a label
    when the text genuinely cannot be classified.
    """
    col = normalize_column(column)

    common_instructions = (
        "The input may contain Telugu-English code-mixed text. "
        "Telugu may be written in Telugu script or Romanized Latin "
        "script. Romanized Telugu can have spelling variations, such "
        "such as 'baagundhi', 'bagundi', 'baagundi', and 'bagundhi'. "
        "Interpret the sentence according to its meaning rather than "
        "requiring standardized spelling. "
        "Short sentences, fragments, and informal expressions can "
        "still contain enough information to classify. "
        "Do not abstain merely because the sentence is short, "
        "informal, incomplete, or Romanized. "
        "Choose the best matching available label whenever the text "
        "provides reasonable evidence. "
        "Use abstain only when the text is genuinely unclassifiable "
        "or none of the provided labels can reasonably apply."
    )

    if "sentiment" in col or "emotion" in col:
        instructions = (
            "Identify the sentiment or emotion expressed by the text. "
            "Consider the meaning of Telugu and English words, including "
            "common Romanized Telugu expressions. "
            "For example, 'baagundhi movie' expresses a favorable "
            "opinion because 'baagundhi' means 'is good' in this context. "
            "Select the best matching sentiment label. "
            + common_instructions
        )

    elif (
        "topic" in col
        or "domain" in col
        or "category" in col
    ):
        instructions = (
            "Identify the main topic or subject of the text. "
            "Use the complete sentence and its context where available. "
            "Select the best matching existing topic label. "
            + common_instructions
        )

    else:
        instructions = (
            f"Determine which existing dataset annotation value "
            f"best applies to the text for the '{column}' field. "
            "Choose only from the supplied labels. "
            + common_instructions
        )

    criteria = {
        f"label_{index}": label_description(column, label)
        for index, label in enumerate(labels)
    }

    criteria["abstain"] = (
        "Use only when the text is genuinely unclassifiable or "
        "none of the supplied labels can reasonably apply. "
        "Do not select this option solely because the text is short "
        "or written in Romanized Telugu."
    )

    return {
        "annotation_check": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }


# ---------------------------------------------------------------------
# Yama response parsing
# ---------------------------------------------------------------------

def extract_answer(
    answer: Any,
) -> tuple[str | None, dict[str, float], float | None]:
    """
    Extract selected choice, probabilities, and confidence.

    Supports dictionary-like and model response objects.
    """
    if answer is None:
        return None, {}, None

    if hasattr(answer, "model_dump"):
        answer = answer.model_dump()

    elif hasattr(answer, "dict") and callable(answer.dict):
        answer = answer.dict()

    if isinstance(answer, str):
        return answer.strip() or None, {}, None

    if not isinstance(answer, dict):
        return None, {}, None

    choice = answer.get("choice")

    if choice is None:
        choice = answer.get(
            "selected_choice",
            answer.get("answer"),
        )

    raw_probabilities = (
        answer.get("probabilities") or {}
    )

    probabilities = {}

    try:
        probabilities = {
            str(key): float(value)
            for key, value in raw_probabilities.items()
            if value is not None
        }

    except (AttributeError, TypeError, ValueError):
        probabilities = {}

    confidence = answer.get("confidence")

    try:
        confidence = (
            float(confidence)
            if confidence is not None
            else None
        )

    except (TypeError, ValueError):
        confidence = None

    # If the selected choice is missing, recover it only from a
    # recognized label-choice probability key.
    if choice is None or not str(choice).strip():
        label_keys = [
            key
            for key in probabilities
            if re.fullmatch(r"label_\d+", key)
        ]

        if label_keys:
            choice = max(
                label_keys,
                key=lambda key: probabilities[key],
            )

    return (
        str(choice).strip() if choice is not None else None,
        probabilities,
        confidence,
    )


def is_abstain_choice(
    choice: str | None,
) -> bool:
    """Identify explicit abstention choices."""
    if choice is None:
        return False

    value = str(choice).strip().casefold()

    return value in {
        "abstain",
        "none",
        "no_label",
        "no label",
        "not_applicable",
        "not applicable",
        "cannot_classify",
        "cannot classify",
        "unknown",
    }


# ---------------------------------------------------------------------
# Availability calculation
# ---------------------------------------------------------------------

def evaluate_availability(
    df: pd.DataFrame,
    annotation_columns: list[str],
) -> dict[str, Any]:
    """Calculate annotation availability."""
    total_records = len(df)

    if total_records == 0:
        raise ValueError(
            "Dataset contains no records."
        )

    if not annotation_columns:
        return {
            "total_records": total_records,
            "annotation_columns": [],
            "annotation_cells": 0,
            "valid_annotation_cells": 0,
            "records_with_annotations": 0,
            "availability_percent": 0.0,
            "not_applicable": True,
        }

    valid_mask = df[annotation_columns].apply(
        lambda column: column.map(valid_value)
    )

    valid_cells = int(
        valid_mask.to_numpy().sum()
    )

    possible_cells = (
        total_records * len(annotation_columns)
    )

    records_with_annotations = int(
        valid_mask.any(axis=1).sum()
    )

    return {
        "total_records": total_records,
        "annotation_columns": [
            str(column)
            for column in annotation_columns
        ],
        "annotation_cells": possible_cells,
        "valid_annotation_cells": valid_cells,
        "records_with_annotations": (
            records_with_annotations
        ),
        "availability_percent": round(
            records_with_annotations
            / total_records
            * 100,
            2,
        ),
        "not_applicable": valid_cells == 0,
    }


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_dataset(
    path: Path,
) -> pd.DataFrame:
    """Load supported dataset formats."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        "Supported input formats are .xlsx, .xls, and .csv."
    )


# ---------------------------------------------------------------------
# Yama verification
# ---------------------------------------------------------------------

def run_yama_verification(
    df: pd.DataFrame,
    text_column: str,
    annotation_columns: list[str],
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Verify existing labels using Yama.

    Blank/abstained predictions are not counted as label mismatches.
    They are reported as unverified cells.
    """
    import thx01

    model = thx01.load(MODEL_ID)

    records: list[dict[str, Any]] = []

    column_summaries: dict[
        str,
        dict[str, Any]
    ] = {}

    for column in annotation_columns:
        valid_annotation_mask = (
            df[column].map(valid_value)
            & df[text_column].map(valid_value)
        )

        row_indices = list(
            df.index[valid_annotation_mask]
        )

        if limit is not None:
            row_indices = row_indices[:limit]

        # IMPORTANT:
        # Use all valid labels in the dataset, not only labels
        # appearing in the limited sample. This prevents a sample
        # from accidentally omitting valid classes.
        all_dataset_labels = sorted(
            {
                normalized_value(value)
                for value in df[column]
                if valid_value(value)
            },
            key=lambda value: value.casefold(),
        )

        if not row_indices or not all_dataset_labels:
            column_summaries[str(column)] = {
                "valid_records_evaluated": 0,
                "predicted_count": 0,
                "abstained_count": 0,
                "unparsed_count": 0,
                "prediction_coverage_percent": None,
                "agreement_among_predictions_percent": None,
                "coverage_adjusted_agreement_percent": None,
                "status": "not_applicable",
            }

            continue

        question_schema = question_for_column(
            str(column),
            all_dataset_labels,
        )

        label_lookup = {
            f"label_{index}": label
            for index, label in enumerate(
                all_dataset_labels
            )
        }

        texts = [
            normalized_value(
                df.at[index, text_column]
            )
            for index in row_indices
        ]

        predictions: list[Any] = []

        # Process in chunks to avoid sending too many records
        # to the model at once.
        chunk_size = max(
            batch_size * 4,
            batch_size,
        )

        for start in range(
            0,
            len(texts),
            chunk_size,
        ):
            chunk = texts[
                start : start + chunk_size
            ]

            batch_results = model.predict_batch(
                chunk,
                question_schema,
                batch_size=batch_size,
            )

            if batch_results is None:
                batch_results = []

            predictions.extend(batch_results)

            processed = min(
                start + len(chunk),
                len(texts),
            )

            print(
                f"Yama | {column}: "
                f"{processed:,}/{len(texts):,} "
                "records processed",
                flush=True,
            )

        if len(predictions) != len(texts):
            raise RuntimeError(
                f"Yama returned {len(predictions)} "
                f"responses for {len(texts)} texts "
                f"in annotation column '{column}'. "
                "Stopping to avoid silently misaligning "
                "predictions with dataset rows."
            )

        matched_count = 0
        predicted_count = 0
        abstained_count = 0
        unparsed_count = 0

        probability_sum = 0.0
        probability_count = 0

        column_rows: list[
            dict[str, Any]
        ] = []

        for (
            source_index,
            text,
            result,
        ) in zip(
            row_indices,
            texts,
            predictions,
        ):
            answers = (
                result.get("answers", {})
                if isinstance(result, dict)
                else {}
            )

            answer_object = (
                answers.get("annotation_check")
                if isinstance(answers, dict)
                else None
            )

            (
                answer_choice,
                probabilities,
                confidence,
            ) = extract_answer(answer_object)

            abstained = is_abstain_choice(
                answer_choice
            )

            predicted_label = (
                label_lookup.get(answer_choice)
            )

            supplied_label = normalized_value(
                df.at[source_index, column]
            )

            if predicted_label is not None:
                predicted_count += 1

            elif abstained:
                abstained_count += 1

            else:
                unparsed_count += 1

            # Find the probability assigned to the
            # dataset's original label.
            supplied_choice = next(
                (
                    key
                    for key, label
                    in label_lookup.items()
                    if label.casefold()
                    == supplied_label.casefold()
                ),
                None,
            )

            supplied_probability = (
                probabilities.get(supplied_choice)
                if supplied_choice
                else None
            )

            selected_choice_probability = (
                probabilities.get(answer_choice)
                if answer_choice
                else None
            )

            if supplied_probability is not None:
                probability_sum += (
                    supplied_probability
                )

                probability_count += 1

            # A blank/abstained prediction is unverified.
            # It is not recorded as a label mismatch.
            is_match = (
                predicted_label is not None
                and predicted_label.casefold()
                == supplied_label.casefold()
            )

            if is_match:
                matched_count += 1

            prediction_status = (
                "predicted"
                if predicted_label is not None
                else (
                    "abstained"
                    if abstained
                    else "unparsed"
                )
            )

            row = {
                "source_row_index": int(
                    source_index
                ),
                "text": text,
                "annotation_column": str(
                    column
                ),
                "provided_label": supplied_label,
                "yama_selected_choice": (
                    answer_choice
                ),
                "yama_predicted_label": (
                    predicted_label
                ),
                "prediction_status": (
                    prediction_status
                ),
                "yama_selected_choice_probability": (
                    selected_choice_probability
                ),
                "probability_assigned_to_provided_label": (
                    supplied_probability
                ),
                "yama_answer_confidence": (
                    confidence
                ),
                "exact_label_match": (
                    is_match
                    if predicted_label is not None
                    else None
                ),
                "review_recommended": (
                    predicted_label is not None
                    and not is_match
                ),
            }

            records.append(row)
            column_rows.append(row)

        total_evaluated = len(column_rows)

        prediction_coverage = (
            predicted_count
            / total_evaluated
            * 100
            if total_evaluated
            else None
        )

        agreement = (
            matched_count
            / predicted_count
            * 100
            if predicted_count
            else None
        )

        coverage_adjusted = (
            matched_count
            / total_evaluated
            * 100
            if total_evaluated
            else None
        )

        column_summaries[str(column)] = {
            "valid_records_evaluated": (
                total_evaluated
            ),
            "predicted_count": (
                predicted_count
            ),
            "abstained_count": (
                abstained_count
            ),
            "unparsed_count": (
                unparsed_count
            ),
            "prediction_coverage_percent": (
                round(prediction_coverage, 2)
                if prediction_coverage is not None
                else None
            ),
            "model_label_agreement_count": (
                matched_count
            ),
            "agreement_among_predictions_percent": (
                round(agreement, 2)
                if agreement is not None
                else None
            ),
            "coverage_adjusted_agreement_percent": (
                round(coverage_adjusted, 2)
                if coverage_adjusted is not None
                else None
            ),
            "mean_probability_assigned_to_dataset_label": (
                round(
                    probability_sum
                    / probability_count,
                    4,
                )
                if probability_count
                else None
            ),
            "status": (
                "completed"
                if total_evaluated
                else "not_applicable"
            ),
            "labels_considered": (
                all_dataset_labels
            ),
        }

        checkpoint_path = (
            output_dir
            / f"metric_04_{normalize_column(column)}_checkpoint.csv"
        )

        pd.DataFrame(
            column_rows
        ).to_csv(
            checkpoint_path,
            index=False,
            encoding="utf-8-sig",
        )

    # -----------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------

    evaluated_count = sum(
        summary["valid_records_evaluated"]
        for summary in column_summaries.values()
    )

    predicted_count = sum(
        summary["predicted_count"]
        for summary in column_summaries.values()
    )

    abstained_count = sum(
        summary["abstained_count"]
        for summary in column_summaries.values()
    )

    unparsed_count = sum(
        summary["unparsed_count"]
        for summary in column_summaries.values()
    )

    agreement_count = sum(
        summary["model_label_agreement_count"]
        for summary in column_summaries.values()
    )

    prediction_coverage = (
        predicted_count
        / evaluated_count
        * 100
        if evaluated_count
        else None
    )

    agreement_among_predictions = (
        agreement_count
        / predicted_count
        * 100
        if predicted_count
        else None
    )

    coverage_adjusted_reliability = (
        agreement_count
        / evaluated_count
        * 100
        if evaluated_count
        else None
    )

    # Weighted mean of Yama's probability assigned to
    # the original dataset labels.
    weighted_probabilities = [
        (
            summary[
                "mean_probability_assigned_to_dataset_label"
            ],
            summary[
                "valid_records_evaluated"
            ],
        )
        for summary in column_summaries.values()
        if summary[
            "mean_probability_assigned_to_dataset_label"
        ] is not None
    ]

    mean_probability = (
        round(
            sum(
                score * count
                for score, count
                in weighted_probabilities
            )
            / sum(
                count
                for _, count
                in weighted_probabilities
            ),
            4,
        )
        if weighted_probabilities
        else None
    )

    result = {
        "metric": (
            "Metric 04 - Annotation Reliability"
        ),
        "metric_version": (
            "Yama-assisted V3 - prompt-refined, "
            "abstention-aware"
        ),
        "status": (
            "completed"
            if evaluated_count
            else "not_applicable"
        ),
        "model_id": MODEL_ID,
        "method": (
            "Yama zero-shot label prediction "
            "compared with existing dataset annotations."
        ),
        "total_records": len(df),
        "text_column": text_column,
        "availability": evaluate_availability(
            df,
            annotation_columns,
        ),
        "annotation_columns": (
            column_summaries
        ),
        "annotation_cells_evaluated": (
            evaluated_count
        ),
        "yama_predicted_cells": (
            predicted_count
        ),
        "yama_abstained_cells": (
            abstained_count
        ),
        "yama_unparsed_cells": (
            unparsed_count
        ),
        "yama_prediction_coverage_percent": (
            round(prediction_coverage, 2)
            if prediction_coverage is not None
            else None
        ),
        "agreement_among_nonblank_predictions_percent": (
            round(
                agreement_among_predictions,
                2,
            )
            if agreement_among_predictions is not None
            else None
        ),
        "coverage_adjusted_reliability_score": (
            round(
                coverage_adjusted_reliability,
                2,
            )
            if coverage_adjusted_reliability is not None
            else None
        ),
        "annotation_reliability_score": (
            round(
                coverage_adjusted_reliability,
                2,
            )
            if coverage_adjusted_reliability is not None
            else 0.0
        ),
        "mean_probability_assigned_to_provided_labels": (
            mean_probability
        ),
        "sample_only": (
            limit is not None
        ),
        "overall_score_inclusion": (
            evaluated_count > 0
            and limit is None
        ),
        "overall_score_exclusion_reason": (
            "Sample-only run; exclude from final aggregation."
            if limit is not None
            else (
                None
                if evaluated_count > 0
                else "No valid annotation cells were evaluated."
            )
        ),
        "interpretation": (
            "Model-based verification, not human ground truth. "
            "Prediction coverage is reported separately from "
            "agreement among predictions. The reliability score "
            "is coverage-adjusted: confirmed matches divided by "
            "all valid annotation cells. Abstained and unparsed "
            "cells remain unverified. Source labels are never changed."
        ),
        "limit_per_annotation_column": limit,
        "evaluation_scope": (
            f"first {limit} annotated records per column "
            "(sample run)"
            if limit is not None
            else "all valid annotated records"
        ),
    }

    # Save detailed record-level report.
    record_df = pd.DataFrame(records)

    record_df.to_csv(
        output_dir
        / "metric_04_yama_record_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result, record_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--text-column",
        default=None,
    )

    parser.add_argument(
        "--annotation-columns",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N annotated records "
            "per column. Sample runs are excluded from "
            "the overall score."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be at least 1"
        )

    if args.limit is not None and args.limit < 1:
        parser.error(
            "--limit must be at least 1"
        )

    df = load_dataset(args.input)

    annotation_columns = detect_annotation_columns(
        df,
        args.annotation_columns,
    )

    text_column = detect_text_column(
        df,
        args.text_column,
    )

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    availability = evaluate_availability(
        df,
        annotation_columns,
    )

    # No applicable annotations:
    # score 0 and exclude from overall aggregation.
    if availability["not_applicable"]:
        result = {
            "metric": (
                "Metric 04 - Annotation Reliability"
            ),
            "metric_version": (
                "Yama-assisted V3"
            ),
            "status": "not_applicable",
            "model_id": MODEL_ID,
            "availability": availability,
            "annotation_reliability_score": 0.0,
            "overall_score_inclusion": False,
            "overall_score_exclusion_reason": (
                "No valid annotation values were found."
            ),
        }

        summary_path = (
            output_dir
            / "metric_04_annotation_reliability.json"
        )

        summary_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            "No valid annotations found: "
            "score 0, excluded from aggregate."
        )

        return

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        "Detected annotation columns: "
        f"{annotation_columns}"
    )

    print(
        "Annotation availability: "
        f"{availability['availability_percent']:.2f}%"
    )

    print(
        f"Loading Yama model: {MODEL_ID}"
    )

    print(
        "First model use may download approximately "
        "1.2 GB of weights.",
        flush=True,
    )

    result, _ = run_yama_verification(
        df=df,
        text_column=text_column,
        annotation_columns=annotation_columns,
        output_dir=output_dir,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    result["input_file"] = str(
        args.input
    )

    summary_path = (
        output_dir
        / "metric_04_annotation_reliability.json"
    )

    summary_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nMETRIC 04 - ANNOTATION RELIABILITY"
    )

    print("-" * 46)

    print(
        f"Records: {result['total_records']:,}"
    )

    print(
        "Annotation columns: "
        f"{annotation_columns}"
    )

    print(
        "Annotation cells evaluated: "
        f"{result['annotation_cells_evaluated']:,}"
    )

    print(
        "Yama prediction coverage: "
        f"{result['yama_prediction_coverage_percent']}%"
    )

    print(
        "Agreement among predictions: "
        f"{result['agreement_among_nonblank_predictions_percent']}%"
    )

    print(
        "Coverage-adjusted reliability: "
        f"{result['annotation_reliability_score']}%"
    )

    print(
        "Yama abstentions: "
        f"{result['yama_abstained_cells']:,}"
    )

    print(
        "Unparsed responses: "
        f"{result['yama_unparsed_cells']:,}"
    )

    print(
        f"Status: {result['status']}"
    )

    print(
        "Included in overall score: "
        f"{result['overall_score_inclusion']}"
    )

    print(
        f"Reports saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
