"""
Metric 04 - Annotation Reliability (Yama zero-shot verification)

Stage 1 detects annotation columns and computes availability.
Stage 2 asks Yama to independently classify the text using the dataset's
existing label set, then compares its prediction with the supplied label.

Interpretation: agreement with Yama is a model-based reliability proxy, not
human-verified ground truth. Do not use this metric as proof that labels are
factually correct.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "metric_04"

MODEL_ID = "LatentAILabs/yama-indic"
DEFAULT_BATCH_SIZE = 16

ANNOTATION_PATTERNS = (
    r"labels?", r"annotations?", r"targets?", r"classes?", r"category",
    r"sentiment", r"intents?", r"tags?", r"ner", r"entity_?labels?",
    r"topics?", r"rating", r"answers?", r"ground_truth", r"gold_?labels?",
)
EXCLUDED_PATTERNS = (
    r"id", r".*_id", r"text", r"sentence", r"utterance", r"content",
    r"review", r"comment", r"source", r"dataset", r"split", r"partition",
    r"script", r"language", r"annotation_method", r"annotator(_id)?",
    r"created_at", r"updated_at", r"timestamp", r"url",
)
TEXT_COLUMN_CANDIDATES = (
    "text", "sentence", "utterance", "message", "content", "review", "comment"
)
MISSING_LABELS = {"", "nan", "none", "null", "n/a"}


def normalize_column(name: Any) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[\-/]+", "_", value)
    return re.sub(r"\s+", "_", value)


def normalized_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def valid_value(value: Any) -> bool:
    return normalized_value(value).casefold() not in MISSING_LABELS


def detect_annotation_columns(
    df: pd.DataFrame, explicit: list[str] | None = None
) -> list[str]:
    if explicit:
        missing = [column for column in explicit if column not in df.columns]
        if missing:
            raise ValueError(f"Annotation columns not found: {missing}")
        return explicit

    detected = []
    for column in df.columns:
        name = normalize_column(column)
        if any(re.fullmatch(pattern, name) for pattern in EXCLUDED_PATTERNS):
            continue
        if any(re.fullmatch(pattern, name) for pattern in ANNOTATION_PATTERNS):
            detected.append(column)
    return detected


def detect_text_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Text column '{explicit}' not found.")
        return explicit
    normalized = {normalize_column(col): col for col in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "Could not identify the text column. Specify it with --text-column."
    )


def label_description(column: str, label: str) -> str:
    """Create a small plain-English explanation for Yama's option."""
    col = normalize_column(column)
    key = label.strip().casefold()

    if "sentiment" in col or "emotion" in col:
        aliases = {
            "positive": "positive sentiment, favorable or happy",
            "pos": "positive sentiment, favorable or happy",
            "negative": "negative sentiment, unfavorable or unhappy",
            "neg": "negative sentiment, unfavorable or unhappy",
            "neutral": "neutral sentiment, neither positive nor negative",
            "mixed": "mixed positive and negative sentiment",
        }
        if key in aliases:
            return f"Sentiment label '{label}': {aliases[key]}."
        return f"Sentiment or emotion category: {label}."

    if "topic" in col or "domain" in col or "category" in col:
        return f"Topic/category label '{label}'; the text is mainly about this subject."

    return f"The dataset's annotation value '{label}' for the '{column}' field."


def question_for_column(column: str, labels: list[str]) -> dict:
    col = normalize_column(column)
    if "sentiment" in col or "emotion" in col:
        instructions = (
            "What sentiment or emotion is expressed by this Telugu-English or "
            "other code-mixed text? Select the best matching label from the "
            "provided dataset labels. If none fits, choose the abstain option."
        )
    elif "topic" in col or "domain" in col or "category" in col:
        instructions = (
            "What is the main topic or subject of this text? Select the best "
            "matching existing dataset label. If none fits, choose the abstain option."
        )
    else:
        instructions = (
            f"What value from the dataset's '{column}' annotation field best "
            "applies to this text? Choose only from the provided labels. "
            "If none fits, choose the abstain option."
        )

    criteria = {
        f"label_{index}": label_description(column, label)
        for index, label in enumerate(labels)
    }
    criteria["abstain"] = "None of the provided labels can reasonably describe this text."
    return {
        "annotation_check": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }


def extract_answer(answer: Any) -> tuple[str | None, dict[str, float], float | None]:
    """Support thx01 answer objects and dict-like serialized responses."""
    if answer is None:
        return None, {}, None

    if hasattr(answer, "model_dump"):
        answer = answer.model_dump()
    elif hasattr(answer, "dict") and callable(answer.dict):
        answer = answer.dict()

    if not isinstance(answer, dict):
        return None, {}, None

    choice = answer.get("choice")
    probabilities = answer.get("probabilities") or {}
    try:
        probabilities = {str(k): float(v) for k, v in probabilities.items()}
    except (AttributeError, TypeError, ValueError):
        probabilities = {}

    confidence = answer.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return (str(choice) if choice is not None else None), probabilities, confidence


def evaluate_availability(
    df: pd.DataFrame, annotation_columns: list[str]
) -> dict[str, Any]:
    total = len(df)
    if total == 0:
        raise ValueError("Dataset contains no records.")

    if not annotation_columns:
        return {
            "total_records": total,
            "annotation_columns": [],
            "annotation_cells": 0,
            "valid_annotation_cells": 0,
            "records_with_annotations": 0,
            "availability_percent": 0.0,
            "not_applicable": True,
        }

    valid = df[annotation_columns].apply(lambda col: col.map(valid_value))
    valid_cells = int(valid.to_numpy().sum())
    possible_cells = int(total * len(annotation_columns))
    records_with = int(valid.any(axis=1).sum())
    return {
        "total_records": total,
        "annotation_columns": [str(c) for c in annotation_columns],
        "annotation_cells": possible_cells,
        "valid_annotation_cells": valid_cells,
        "records_with_annotations": records_with,
        "availability_percent": round(records_with / total * 100, 2),
        "not_applicable": valid_cells == 0,
    }


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Supported input formats are .xlsx, .xls, and .csv")


def run_yama_verification(
    df: pd.DataFrame,
    text_column: str,
    annotation_columns: list[str],
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run Yama on valid annotation cells; writes progress/checkpoint CSV."""
    import thx01  # Import only when annotation labels exist and verification is needed.

    model = thx01.load(MODEL_ID)
    records: list[dict[str, Any]] = []
    col_summary: dict[str, dict[str, Any]] = {}

    for column in annotation_columns:
        mask = df[column].map(valid_value) & df[text_column].map(valid_value)
        row_indices = list(df.index[mask])
        if limit is not None:
            row_indices = row_indices[:limit]

        labels = sorted({normalized_value(df.at[i, column]) for i in row_indices})
        if not labels:
            col_summary[str(column)] = {
                "valid_records_evaluated": 0,
                "agreement_percent": None,
                "mean_dataset_label_probability": None,
                "status": "not_applicable",
            }
            continue

        question_schema = question_for_column(str(column), labels)
        label_lookup = {
            f"label_{index}": label for index, label in enumerate(labels)
        }
        states = [normalized_value(df.at[i, text_column]) for i in row_indices]

        predictions: list[dict] = []
        # Keep inference chunks bounded; each chunk can be restarted manually if interrupted.
        chunk_size = max(batch_size * 4, batch_size)
        for start in range(0, len(states), chunk_size):
            chunk_states = states[start : start + chunk_size]
            batch_results = model.predict_batch(
                chunk_states,
                question_schema,
                batch_size=batch_size,
            )
            predictions.extend(batch_results)
            done = min(start + len(chunk_states), len(states))
            print(
                f"Yama | {column}: {done:,}/{len(states):,} records processed",
                flush=True,
            )

        matched_count = 0
        label_probability_sum = 0.0
        probability_count = 0
        column_rows: list[dict[str, Any]] = []

        for source_index, state, result in zip(row_indices, states, predictions):
            answers = result.get("answers", {}) if isinstance(result, dict) else {}
            answer_choice, probabilities, confidence = extract_answer(
                answers.get("annotation_check")
            )
            predicted_label = label_lookup.get(answer_choice, None)
            supplied_label = normalized_value(df.at[source_index, column])
            is_match = (
                predicted_label is not None
                and predicted_label.casefold() == supplied_label.casefold()
            )

            supplied_key = next(
                (
                    key for key, label in label_lookup.items()
                    if label.casefold() == supplied_label.casefold()
                ),
                None,
            )
            supplied_probability = (
                probabilities.get(supplied_key)
                if supplied_key is not None
                else None
            )
            if supplied_probability is not None:
                label_probability_sum += supplied_probability
                probability_count += 1
            if is_match:
                matched_count += 1

            row = {
                "source_row_index": int(source_index),
                "text": state,
                "annotation_column": str(column),
                "provided_label": supplied_label,
                "yama_predicted_label": predicted_label,
                "yama_top_label_probability": (
                    probabilities.get(answer_choice) if answer_choice else None
                ),
                "probability_assigned_to_provided_label": supplied_probability,
                "yama_answer_confidence": confidence,
                "exact_label_match": bool(is_match),
                "review_recommended": not bool(is_match),
            }
            records.append(row)
            column_rows.append(row)

        n = len(column_rows)
        col_summary[str(column)] = {
            "valid_records_evaluated": n,
            "model_label_agreement_count": matched_count,
            "agreement_percent": round(matched_count / n * 100, 2) if n else None,
            "mean_probability_assigned_to_dataset_label": (
                round(label_probability_sum / probability_count, 4)
                if probability_count else None
            ),
            "status": "completed" if n else "not_applicable",
            "labels_considered": labels,
        }

        # Save a per-column checkpoint so a long inference run has tangible output.
        checkpoint = output_dir / f"metric_04_{normalize_column(column)}_checkpoint.csv"
        pd.DataFrame(column_rows).to_csv(
            checkpoint, index=False, encoding="utf-8-sig"
        )

    evaluated = sum(v["valid_records_evaluated"] for v in col_summary.values())
    agreements = sum(v["model_label_agreement_count"] for v in col_summary.values())
    reliability = round(agreements / evaluated * 100, 2) if evaluated else None
    weighted_probability_values = [
        v["mean_probability_assigned_to_dataset_label"] * v["valid_records_evaluated"]
        for v in col_summary.values()
        if v["mean_probability_assigned_to_dataset_label"] is not None
    ]
    weighted_probability_counts = [
        v["valid_records_evaluated"]
        for v in col_summary.values()
        if v["mean_probability_assigned_to_dataset_label"] is not None
    ]
    mean_support = (
        round(
            sum(weighted_probability_values) / sum(weighted_probability_counts),
            4,
        )
        if weighted_probability_counts else None
    )

    result = {
        "metric": "Metric 04 - Annotation Reliability",
        "metric_version": "Yama-assisted V1",
        "status": "completed" if evaluated else "not_applicable",
        "model_id": MODEL_ID,
        "method": "Zero-shot Yama label prediction compared against existing dataset annotations.",
        "total_records": len(df),
        "text_column": text_column,
        "availability": evaluate_availability(df, annotation_columns),
        "annotation_columns": col_summary,
        "annotation_cells_evaluated": evaluated,
        "exact_model_label_agreement_percent": reliability,
        "mean_probability_assigned_to_provided_labels": mean_support,
        "annotation_reliability_score": reliability,
        "sample_only": limit is not None,
        "overall_score_inclusion": evaluated > 0 and limit is None,
        "overall_score_exclusion_reason": (
            "Sample-only run; do not use this score in the final aggregate."
            if limit is not None
            else (None if evaluated > 0 else "No valid annotation cells were evaluated.")
        ),
        "interpretation": (
            "The score is agreement with Yama's zero-shot predictions, not verified "
            "ground-truth accuracy or human inter-annotator agreement. Mismatches are "
            "review candidates; source labels are never modified."
        ),
        "limit_per_annotation_column": limit,
        "evaluation_scope": (
            f"first {limit} annotated records per column (smoke test)"
            if limit is not None else "all valid annotated records"
        ),
    }

    record_df = pd.DataFrame(records)
    record_df.to_csv(
        output_dir / "metric_04_yama_record_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result, record_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--annotation-columns", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N annotated rows per column (smoke test).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    df = load_dataset(args.input)
    columns = detect_annotation_columns(df, args.annotation_columns)
    text_column = detect_text_column(df, args.text_column)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    availability = evaluate_availability(df, columns)
    if availability["not_applicable"]:
        result = {
            "metric": "Metric 04 - Annotation Reliability",
            "metric_version": "Yama-assisted V1",
            "status": "not_applicable",
            "model_id": MODEL_ID,
            "availability": availability,
            "annotation_reliability_score": 0.0,
            "overall_score_inclusion": False,
            "overall_score_exclusion_reason": "No valid annotation values were found.",
        }
        (output_dir / "metric_04_annotation_reliability.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("No valid annotations found: score 0, excluded from aggregate.")
        return

    print(f"Dataset rows: {len(df):,}")
    print(f"Detected annotation columns: {columns}")
    print(f"Annotation availability: {availability['availability_percent']:.2f}%")
    print(f"Loading Yama model: {MODEL_ID}")
    print("First model use may download approximately 1.2 GB of weights.", flush=True)

    result, _ = run_yama_verification(
        df=df,
        text_column=text_column,
        annotation_columns=columns,
        output_dir=output_dir,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    result["input_file"] = str(args.input)
    (output_dir / "metric_04_annotation_reliability.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nMETRIC 04 - ANNOTATION RELIABILITY")
    print("-" * 46)
    print(f"Records: {result['total_records']:,}")
    print(f"Annotation columns: {columns}")
    print(f"Annotation cells evaluated: {result['annotation_cells_evaluated']:,}")
    print(f"Yama agreement: {result['exact_model_label_agreement_percent']}%")
    print(f"Reliability score: {result['annotation_reliability_score']}%")
    print(f"Status: {result['status']}")
    print(f"Included in overall score: {result['overall_score_inclusion']}")
    print(f"Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
