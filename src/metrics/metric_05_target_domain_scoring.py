"""
Metric 05 - Target-Domain Relevance Scoring (Prototype V1)

This is a separate scoring layer. It does NOT modify the frozen V2.1
unsupervised semantic-domain discovery script.

What it measures:
    The percentage of records whose multilingual embedding similarity to a
    user-defined target-domain description meets a configurable threshold.

Important:
    This is a provisional semantic-relevance proxy, not proof of population
    representativeness. The similarity threshold must be inspected/calibrated
    against reviewed examples before the score is treated as validated.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.50


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Supported formats are CSV and Excel.")

    if df.empty:
        raise ValueError("The dataset is empty.")
    return df


def identify_text_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Column '{requested}' not found. Available columns: {list(df.columns)}"
            )
        return requested

    for candidate in (
        "text", "content", "sentence", "review", "comment",
        "message", "description", "tweet", "post", "body",
    ):
        for column in df.columns:
            if str(column).strip().lower() == candidate:
                return column

    text_columns = {}
    for column in df.columns:
        values = df[column].dropna().astype(str)
        if not values.empty:
            text_columns[column] = values.str.len().mean()

    if not text_columns:
        raise ValueError("No usable text column found.")

    return max(text_columns, key=text_columns.get)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def encode_texts(model, texts, prefix: str) -> np.ndarray:
    prepared = [f"{prefix}: {text}" for text in texts]
    return model.encode(
        prepared,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def ask_domain() -> tuple[str, str]:
    print("\nMetric 05: Target-Domain Relevance")
    domain = input(
        "Enter the intended dataset domain (e.g., Healthcare, Education, "
        "Telecommunications). Press Enter for General: "
    ).strip()
    if not domain:
        domain = "General"

    description = input(
        "Briefly describe what counts as relevant to this domain "
        "(recommended; press Enter to use a basic domain description): "
    ).strip()

    if not description:
        if domain.casefold() == "general":
            description = (
                "A general-purpose Telugu-English code-mixed dataset covering "
                "everyday subjects, experiences, opinions, and conversations."
            )
        else:
            description = (
                f"Telugu-English code-mixed text about the domain of {domain}, "
                f"including its concepts, activities, terminology, and user experiences."
            )

    return domain, description


def ask_threshold(default: float) -> float:
    raw = input(
        f"Similarity threshold from 0 to 1 [provisional default {default:.2f}]: "
    ).strip()
    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("Threshold must be a number between 0 and 1.") from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    return value


def score_target_domain(
    df: pd.DataFrame,
    text_column: str,
    domain: str,
    domain_description: str,
    threshold: float,
) -> tuple[pd.DataFrame, dict]:
    records = df.copy()
    records["processed_text"] = records[text_column].apply(clean_text)

    valid_mask = records["processed_text"].str.len() > 0
    valid = records.loc[valid_mask].copy()

    if valid.empty:
        raise ValueError("No non-empty text records were found.")

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # E5 retrieval-style format: domain prompt as query, dataset texts as passages.
    domain_vector = encode_texts(model, [domain_description], "query")
    record_vectors = encode_texts(
        model, valid["processed_text"].tolist(), "passage"
    )

    similarities = cosine_similarity(record_vectors, domain_vector).ravel()

    valid["target_domain_similarity"] = similarities
    valid["target_domain_relevant"] = similarities >= threshold
    valid["target_domain"] = domain

    output = records.copy()
    output["target_domain"] = domain
    output["target_domain_similarity"] = np.nan
    output["target_domain_relevant"] = pd.NA

    output.loc[valid.index, "target_domain_similarity"] = valid[
        "target_domain_similarity"
    ]
    output.loc[valid.index, "target_domain_relevant"] = valid[
        "target_domain_relevant"
    ].astype(bool)

    relevant_count = int(valid["target_domain_relevant"].sum())
    evaluated_count = int(len(valid))
    coverage = (relevant_count / evaluated_count * 100.0) if evaluated_count else 0.0

    report = {
        "metric": "Metric 05 - Target-Domain Relevance Scoring",
        "version": "PROTOTYPE-V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_domain": domain,
        "target_domain_description": domain_description,
        "similarity_model": MODEL_NAME,
        "similarity_method": "Cosine similarity of normalized multilingual E5 embeddings",
        "classification_threshold": threshold,
        "threshold_status": "Provisional; requires review/calibration on sampled records",
        "dataset": {
            "total_records": int(len(records)),
            "evaluated_records": evaluated_count,
            "empty_text_records": int(len(records) - evaluated_count),
        },
        "domain_relevance": {
            "relevant_records": relevant_count,
            "not_relevant_records": int(evaluated_count - relevant_count),
            "target_domain_coverage_score": round(coverage, 2),
        },
        "interpretation": (
            "The score estimates the share of records semantically relevant to "
            "the user-declared domain under the selected similarity threshold. "
            "It is an internal relevance/coverage proxy, not proof of population "
            "representativeness or complete subdomain coverage."
        ),
        "limitations": [
            "The threshold is provisional and must be checked against human-reviewed examples.",
            "Embedding similarity may be affected by short text, slang, transliteration, dialect, and code-mixing.",
            "A record can be relevant to more than one domain; this prototype evaluates one target domain per run.",
            "The score does not establish representativeness of a real-world population without a reference distribution.",
            "This prototype recomputes embeddings separately from frozen V2.1; later integration can reuse embeddings.",
        ],
    }

    return output, report


def main():
    parser = argparse.ArgumentParser(
        description="Prototype target-domain relevance scoring for TelCo-QA Metric 05."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV or Excel dataset.",
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument(
        "--domain",
        default=None,
        help="Target domain. If omitted, the script asks interactively.",
    )
    parser.add_argument(
        "--domain-description",
        default=None,
        help="Detailed description of what counts as relevant to the target domain.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Cosine similarity threshold (provisional default {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args()

    if args.domain:
        domain = args.domain.strip()
        description = (args.domain_description or "").strip()
        if not description:
            description = (
                f"Telugu-English code-mixed text relevant to the domain of {domain}, "
                "including its concepts, activities, terminology, and user experiences."
            )
    else:
        domain, prompted_description = ask_domain()
        description = args.domain_description.strip() if args.domain_description else prompted_description

    threshold = args.threshold if args.threshold is not None else ask_threshold(
        DEFAULT_THRESHOLD
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    data_path = Path(args.dataset).expanduser()
    df = load_dataset(data_path)
    text_column = identify_text_column(df, args.text_column)

    print(f"Dataset: {data_path.name}")
    print(f"Text column: {text_column}")
    print(f"Target domain: {domain}")
    print(f"Threshold: {threshold:.3f}")

    record_output, report = score_target_domain(
        df=df,
        text_column=text_column,
        domain=domain,
        domain_description=description,
        threshold=threshold,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_domain = re.sub(r"[^A-Za-z0-9_-]+", "_", domain.strip()).strip("_") or "domain"
    records_path = OUTPUT_DIR / f"metric_05_target_domain_{safe_domain}_records.csv"
    report_path = OUTPUT_DIR / f"metric_05_target_domain_{safe_domain}_report.json"

    record_output.to_csv(records_path, index=False, encoding="utf-8-sig")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    result = report["domain_relevance"]
    print("\n" + "=" * 60)
    print("METRIC 05 - TARGET-DOMAIN RELEVANCE")
    print("=" * 60)
    print(f"Target domain: {domain}")
    print(f"Relevant records: {result['relevant_records']}")
    print(f"Evaluated records: {report['dataset']['evaluated_records']}")
    print(f"Domain coverage score: {result['target_domain_coverage_score']}%")
    print(f"Threshold: {threshold:.3f} (provisional)")
    print("\nInterpretation:")
    print(report["interpretation"])
    print("\nOutputs:")
    print(f"Record-level CSV: {records_path}")
    print(f"JSON report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
