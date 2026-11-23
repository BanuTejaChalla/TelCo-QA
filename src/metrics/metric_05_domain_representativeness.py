"""
Metric 05 - Zero-Shot Target-Domain Classification

Classifies every non-empty Telugu-English record into one of the dataset's
seven domain categories using a pretrained multilingual zero-shot model.

No local classifier training is required. The pretrained model is downloaded
from Hugging Face the first time and cached for later runs.

Run from the TelCo-QA repository root:
    python src/metrics/metric_05_domain_representativeness.py

Optional:
    python src/metrics/metric_05_domain_representativeness.py --domain "Cricket & Sports"
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from transformers import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "Horizon-Labs/multilingual-zeroshot-small"
TEXT_COLUMN = "text"
LABEL_COLUMN = "topic"

# Candidate labels are deliberately descriptive, not just short category names.
# Zero-shot models often respond better to clear, natural-language labels.
DOMAIN_DESCRIPTIONS = {
    "General": (
        "general everyday conversation, personal experiences, casual opinions, "
        "or statements not mainly about another specific listed domain"
    ),
    "Cricket & Sports": (
        "cricket or other sports, matches, batting, bowling, players, teams, "
        "tournaments, scores, sports news, sports performance, or fan opinions"
    ),
    "Education & Students": (
        "education, schools, colleges, students, exams, studying, teachers, "
        "courses, academic life, or learning"
    ),
    "Technology & AI": (
        "technology, computers, software, programming, artificial intelligence, "
        "apps, digital products, gadgets, or technical topics"
    ),
    "Politics & Current Affairs": (
        "politics, government, political leaders, elections, public policy, "
        "or current affairs"
    ),
    "Movies & Entertainment": (
        "movies, films, cinema, actors, acting, directors, songs, television, "
        "celebrity entertainment, or reviews of entertainment"
    ),
    "Multiple topics": (
        "a sentence that meaningfully discusses two or more different domains "
        "from this list rather than having one main domain"
    ),
}

# Keep the order consistent in the output and reports.
DOMAIN_NAMES = list(DOMAIN_DESCRIPTIONS.keys())
CANDIDATE_LABELS = [
    f"{name}: {description}"
    for name, description in DOMAIN_DESCRIPTIONS.items()
]
LABEL_TO_DOMAIN = {
    candidate: domain
    for candidate, domain in zip(CANDIDATE_LABELS, DOMAIN_NAMES)
}


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


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def classify_records(texts, classifier, batch_size: int):
    """Classify texts in batches and return domain, score, and full score map."""
    predicted_domains = []
    top_scores = []
    all_scores = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        results = classifier(
            batch,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template="This Telugu-English sentence is mainly about {}.",
            multi_label=False,
            batch_size=batch_size,
        )

        # Transformers returns a dict for a single string and a list for a batch.
        if isinstance(results, dict):
            results = [results]

        for result in results:
            ranked_labels = result["labels"]
            ranked_scores = result["scores"]

            domain = LABEL_TO_DOMAIN[ranked_labels[0]]
            score = float(ranked_scores[0])

            score_map = {
                LABEL_TO_DOMAIN[label]: float(value)
                for label, value in zip(ranked_labels, ranked_scores)
            }

            predicted_domains.append(domain)
            top_scores.append(score)
            all_scores.append(score_map)

        completed = min(start + len(batch), len(texts))
        print(f"Classified {completed:,} / {len(texts):,} records")

    return predicted_domains, top_scores, all_scores


def main():
    parser = argparse.ArgumentParser(
        description="Metric 05: Zero-shot domain classification and representation."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV or Excel dataset.",
    )
    parser.add_argument(
        "--domain",
        default="Cricket & Sports",
        help="Target domain whose share should be reported.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size. Lower it if memory is limited.",
    )
    args = parser.parse_args()

    target_domain = args.domain.strip()
    if target_domain not in DOMAIN_DESCRIPTIONS:
        raise ValueError(
            f"Unknown target domain: {target_domain}\n"
            f"Choose one of: {', '.join(DOMAIN_NAMES)}"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    dataset_path = Path(args.dataset).expanduser()
    df = load_dataset(dataset_path)

    if TEXT_COLUMN not in df.columns:
        raise KeyError(
            f"Required text column '{TEXT_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    records = df.copy()
    records["processed_text"] = records[TEXT_COLUMN].apply(clean_text)
    valid_mask = records["processed_text"].str.len() > 0
    valid = records.loc[valid_mask].copy()

    if valid.empty:
        raise ValueError("No non-empty text records were found.")

    print(f"Dataset: {dataset_path}")
    print(f"Records: {len(records):,}")
    print(f"Non-empty records: {len(valid):,}")
    print(f"Target domain: {target_domain}")
    print(f"Loading zero-shot model: {MODEL_NAME}")
    print("The model is downloaded once and cached locally.")

    classifier = pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME,
        device=-1,  # CPU; works without requiring a CUDA setup
    )

    predictions, confidence_scores, score_maps = classify_records(
        valid["processed_text"].tolist(),
        classifier,
        args.batch_size,
    )

    valid["predicted_domain"] = predictions
    valid["domain_prediction_confidence"] = confidence_scores

    # Store each category's zero-shot score for later inspection.
    for domain in DOMAIN_NAMES:
        valid[f"zs_score_{domain}"] = [
            scores.get(domain, 0.0) for scores in score_maps
        ]

    # Target relevance is now based on the predicted domain, not cosine
    # similarity or an arbitrary E5 similarity threshold.
    valid["target_domain"] = target_domain
    valid["target_domain_relevant"] = (
        valid["predicted_domain"] == target_domain
    )

    output = records.copy()
    output["target_domain"] = target_domain
    output["predicted_domain"] = pd.NA
    output["domain_prediction_confidence"] = pd.NA
    output["target_domain_relevant"] = pd.NA

    for domain in DOMAIN_NAMES:
        output[f"zs_score_{domain}"] = pd.NA

    output.loc[valid.index, "predicted_domain"] = valid["predicted_domain"]
    output.loc[valid.index, "domain_prediction_confidence"] = (
        valid["domain_prediction_confidence"]
    )
    output.loc[valid.index, "target_domain_relevant"] = (
        valid["target_domain_relevant"].astype(bool)
    )

    for domain in DOMAIN_NAMES:
        output.loc[valid.index, f"zs_score_{domain}"] = valid[
            f"zs_score_{domain}"
        ]

    evaluated_count = len(valid)
    relevant_count = int(valid["target_domain_relevant"].sum())
    target_share = relevant_count / evaluated_count * 100.0

    predicted_distribution = (
        valid["predicted_domain"]
        .value_counts()
        .reindex(DOMAIN_NAMES, fill_value=0)
    )

    # Compare model predictions to dataset topics when the topic column exists.
    label_comparison = None
    if LABEL_COLUMN in valid.columns:
        actual = valid[LABEL_COLUMN].fillna("").astype(str).str.strip()
        usable_labels = actual != ""
        if usable_labels.any():
            agreement = (
                valid.loc[usable_labels, "predicted_domain"]
                == actual.loc[usable_labels]
            ).mean()
            label_comparison = {
                "records_with_existing_topic": int(usable_labels.sum()),
                "prediction_topic_agreement_percent": round(
                    float(agreement) * 100.0, 2
                ),
                "note": (
                    "Agreement with existing topic labels is a diagnostic only; "
                    "it is not an independent accuracy measure."
                ),
            }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_domain = (
        re.sub(r"[^A-Za-z0-9_-]+", "_", target_domain).strip("_") or "domain"
    )
    records_path = (
        OUTPUT_DIR / f"metric_05_zero_shot_{safe_domain}_records.csv"
    )
    report_path = (
        OUTPUT_DIR / f"metric_05_zero_shot_{safe_domain}_report.json"
    )

    output.to_csv(records_path, index=False, encoding="utf-8-sig")

    report = {
        "metric": "Metric 05 - Target-Domain Representativeness",
        "version": "ZERO-SHOT-PROTOTYPE-V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "method": (
            "Multilingual zero-shot sequence classification using descriptive "
            "domain candidate labels; single-label prediction."
        ),
        "target_domain": target_domain,
        "dataset": {
            "total_records": int(len(records)),
            "evaluated_records": int(evaluated_count),
            "empty_text_records": int(len(records) - evaluated_count),
        },
        "target_domain_relevance": {
            "records_predicted_in_target_domain": int(relevant_count),
            "records_predicted_outside_target_domain": int(
                evaluated_count - relevant_count
            ),
            "target_domain_share_percent": round(float(target_share), 2),
        },
        "predicted_domain_distribution": {
            domain: {
                "records": int(predicted_distribution[domain]),
                "percentage": round(
                    float(predicted_distribution[domain] / evaluated_count * 100),
                    2,
                ),
            }
            for domain in DOMAIN_NAMES
        },
        "existing_label_comparison": label_comparison,
        "interpretation": (
            "The target-domain share estimates the proportion of records the "
            "zero-shot model assigns to the selected domain. It is a model-based "
            "domain composition estimate, not proof of population "
            "representativeness. True representativeness requires a defined "
            "reference population or expected domain distribution."
        ),
        "limitations": [
            "Zero-shot scores are model scores, not calibrated probabilities of correctness.",
            "Romanized Telugu and informal code-mixing may not be well represented in pretraining.",
            "Single-label classification assigns one main domain even when a record touches multiple domains.",
            "The model may be sensitive to candidate-label wording and hypothesis template.",
            "Review a sample of predictions before treating the metric as validated.",
        ],
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 62)
    print("METRIC 05 - ZERO-SHOT DOMAIN REPRESENTATIVENESS")
    print("=" * 62)
    print(f"Model: {MODEL_NAME}")
    print(f"Target domain: {target_domain}")
    print(f"Records assigned to target domain: {relevant_count:,}")
    print(f"Evaluated records: {evaluated_count:,}")
    print(f"Target-domain share: {target_share:.2f}%")
    print("\nPredicted domain distribution:")
    for domain in DOMAIN_NAMES:
        count = int(predicted_distribution[domain])
        percentage = count / evaluated_count * 100.0
        print(f"  {domain:<30} {count:>5,} ({percentage:>6.2f}%)")

    if label_comparison:
        print(
            "\nAgreement with existing topic labels: "
            f"{label_comparison['prediction_topic_agreement_percent']:.2f}%"
        )
        print("(Diagnostic only; not independent ground-truth accuracy.)")

    print("\nSaved:")
    print(f"  Records: {records_path}")
    print(f"  Report:  {report_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
