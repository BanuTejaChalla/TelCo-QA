"""
Metric 05: Unsupervised Semantic Domain Discovery — FROZEN V2.1

Purpose:
    Describe the subject-matter structure of a text dataset without
    requiring topic labels, a fixed domain taxonomy, or manual annotations.

Method:
    1. Load CSV/Excel input and identify a text column.
    2. Generate multilingual sentence embeddings.
    3. Reduce embedding dimensionality for density-based clustering.
    4. Discover semantic groups with HDBSCAN.
    5. Describe each group from its own corpus using class-based TF-IDF.
    6. Measure coverage, cluster cohesion, separation, diversity and balance.
    7. Export record-level assignments and a JSON report.

Important:
    This is an unsupervised structural analysis. It does not establish
    representativeness of an intended population, nor does it claim that
    automatically generated topic descriptions are ground-truth labels.
"""

import argparse
import hashlib
import json
import math
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import sentence_transformers
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


# Generic Unicode word boundaries can split Telugu combining marks into
# meaningless character fragments. This tokenizer keeps Telugu-block runs
# together while also supporting Latin words and numbers.
MULTILINGUAL_TOKEN_PATTERN = re.compile(
    r"[\u0C00-\u0C7F]+|[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+"
)


def multilingual_tokenize(text):
    return MULTILINGUAL_TOKEN_PATTERN.findall(text.casefold())
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RECORDS_OUTPUT = OUTPUT_DIR / "metric_05_domain_discovery_records.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "metric_05_domain_discovery.json"

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32
TOP_TERMS_PER_CLUSTER = 8
REPRESENTATIVE_RECORDS = 3
SILHOUETTE_SAMPLE_SIZE = 2000
RANDOM_STATE = 42


# Frozen identity and reproducibility helpers.
FROZEN_VERSION = "V2.1-FROZEN"
FROZEN_METHOD_ID = "telcoqa.metric05.semantic-domain-discovery.v2.1"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata():
    from importlib.metadata import PackageNotFoundError, version

    def package_version(package_name):
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "not-installed"

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "sentence_transformers_version": package_version("sentence-transformers"),
        "umap_learn_version": package_version("umap-learn"),
        "hdbscan_version": package_version("hdbscan"),
    }

# HDBSCAN parameters are derived from dataset size unless overridden.
MIN_CLUSTER_SIZE_FLOOR = 10
MIN_CLUSTER_SIZE_CEILING = 50


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    print(f"\nDataset loaded: {file_path.name}")
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    return df


def identify_text_column(df, requested_column=None):
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(
                f"Text column '{requested_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested_column

    candidates = (
        "text", "content", "sentence", "review", "comment",
        "message", "description", "tweet", "post", "body"
    )
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in normalized:
            selected = normalized[candidate]
            print(f"Automatically selected text column: {selected}")
            return selected

    # Fall back to the column with the greatest average non-empty text length.
    scores = {}
    for column in df.columns:
        values = df[column].dropna().astype(str)
        if len(values):
            scores[column] = values.str.len().mean()

    if not scores:
        raise ValueError("No usable columns were found.")

    selected = max(scores, key=scores.get)
    print(
        f"No conventional text-column name found. "
        f"Selected '{selected}' by average text length. "
        "Pass --text-column to override."
    )
    return selected


# ============================================================
# TEXT PREPARATION
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    # Preserve all scripts, punctuation and words; normalize whitespace only.
    return re.sub(r"\s+", " ", str(value)).strip()


def prepare_records(df, text_column):
    records = df.copy()
    records["processed_text"] = records[text_column].apply(clean_text)
    valid_mask = records["processed_text"].str.len() > 0
    valid_records = records.loc[valid_mask].copy()

    print("\nText preparation")
    print(f"Total records: {len(records)}")
    print(f"Usable text records: {len(valid_records)}")
    print(f"Empty records: {len(records) - len(valid_records)}")

    if valid_records.empty:
        raise ValueError("No non-empty text records were found.")

    return records, valid_records


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embeddings(texts, model):
    print("\nGenerating multilingual sentence embeddings...")
    # E5 recommends the query prefix for retrieval-style text representations.
    prepared = [f"query: {text}" for text in texts]
    vectors = model.encode(
        prepared,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Embedding matrix: {vectors.shape}")
    return vectors.astype(np.float32)


# ============================================================
# DIMENSIONALITY REDUCTION AND CLUSTER DISCOVERY
# ============================================================

def reduce_embeddings(embeddings):
    n_records = len(embeddings)

    if n_records < 5:
        return embeddings

    n_neighbors = min(15, max(2, n_records - 1))
    n_components = min(5, embeddings.shape[1], n_records - 2)

    try:
        import umap

        print("\nReducing embedding dimensions with UMAP...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=RANDOM_STATE,
            transform_seed=RANDOM_STATE,
            n_jobs=1,
            # V2.1 used UMAP's legacy layout pipeline. Make that choice
            # explicit so a future UMAP default change does not silently
            # alter the frozen clustering results.
            compatibility_layout=True,
        )
        reduced = reducer.fit_transform(embeddings)
        return np.asarray(reduced, dtype=np.float32)

    except ImportError:
        print(
            "\nWARNING: umap-learn is not installed. "
            "Using normalized embeddings directly. "
            "Install with: pip install umap-learn"
        )
        return embeddings


def discover_clusters(embeddings):
    n_records = len(embeddings)

    if n_records < MIN_CLUSTER_SIZE_FLOOR:
        print(
            "\nDataset is too small for reliable density-based discovery. "
            "All records will remain unassigned."
        )
        return np.full(n_records, -1, dtype=int), None

    min_cluster_size = int(
        np.clip(
            round(math.sqrt(n_records) * 0.5),
            MIN_CLUSTER_SIZE_FLOOR,
            MIN_CLUSTER_SIZE_CEILING,
        )
    )
    min_samples = max(2, round(min_cluster_size * 0.25))

    reduced = reduce_embeddings(embeddings)

    print("\nDiscovering semantic groups with HDBSCAN...")
    print(f"min_cluster_size: {min_cluster_size}")
    print(f"min_samples: {min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(reduced)
    strengths = clusterer.probabilities_

    cluster_count = len(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))
    print(f"Groups discovered: {cluster_count}")
    print(f"Unassigned/noise records: {noise_count}")

    return labels, strengths


# ============================================================
# AUTOMATIC GROUP DESCRIPTIONS
# ============================================================

def build_stop_words():
    # English stop words only. Do not remove Telugu or transliterated tokens
    # using an assumed language-specific list.
    return list(ENGLISH_STOP_WORDS)


def describe_clusters(texts, embeddings, labels):
    """
    Build data-derived descriptions for clusters.

    Terms are extracted from the combined text of each cluster and weighted
    by their distinctiveness across clusters (class-based TF-IDF style).
    No predefined subject taxonomy is used.
    """
    cluster_ids = sorted(set(labels) - {-1})
    if not cluster_ids:
        return {}, {}

    cluster_documents = []
    for cluster_id in cluster_ids:
        members = [texts[i] for i, label in enumerate(labels) if label == cluster_id]
        cluster_documents.append(" ".join(members))

    vectorizer = CountVectorizer(
        tokenizer=multilingual_tokenize,
        token_pattern=None,
        preprocessor=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=50000,
        stop_words=build_stop_words(),
        min_df=1,
    )

    try:
        count_matrix = vectorizer.fit_transform(cluster_documents)
    except ValueError:
        count_matrix = None

    descriptions = {}
    details = {}
    feature_names = (
        vectorizer.get_feature_names_out()
        if count_matrix is not None else np.array([])
    )

    if count_matrix is not None and len(feature_names):
        # Class-based TF-IDF: term frequency within a cluster, weighted by
        # inverse frequency across cluster documents.
        # Keep the cluster-by-vocabulary matrix sparse to reduce memory use.
        total_terms = np.asarray(count_matrix.sum(axis=1)).ravel()
        row_scale = 1.0 / np.maximum(total_terms, 1.0)
        tf = count_matrix.multiply(row_scale[:, None])
        document_frequency = np.asarray(
            (count_matrix > 0).sum(axis=0)
        ).ravel()
        idf = np.log(
            1.0 + (len(cluster_ids) / (1.0 + document_frequency))
        )
        ctfidf = tf.multiply(idf).tocsr()

    for row_index, cluster_id in enumerate(cluster_ids):
        indices = np.where(labels == cluster_id)[0]
        cluster_vectors = embeddings[indices]
        centroid = normalize(cluster_vectors.mean(axis=0, keepdims=True))[0]
        similarities = cluster_vectors @ centroid
        representative_order = np.argsort(similarities)[::-1]
        representative_indices = indices[
            representative_order[:REPRESENTATIVE_RECORDS]
        ]

        if count_matrix is not None and len(feature_names):
            row_scores = ctfidf.getrow(row_index).toarray().ravel()
            top_indices = np.argsort(row_scores)[::-1]
            terms = [
                str(feature_names[i])
                for i in top_indices
                if row_scores[i] > 0
            ][:TOP_TERMS_PER_CLUSTER]
        else:
            terms = []

        label = " · ".join(terms[:5]) if terms else "Insufficient distinctive terms"
        cohesion = float(np.mean(similarities))

        descriptions[int(cluster_id)] = label
        details[int(cluster_id)] = {
            "group_id": int(cluster_id),
            "generated_description": label,
            "top_terms": terms,
            "record_count": int(len(indices)),
            "mean_cosine_cohesion": round(cohesion, 4),
            "representative_record_indices": [
                int(i) for i in representative_indices
            ],
            "representative_texts": [
                texts[i] for i in representative_indices
            ],
        }

    return descriptions, details


# ============================================================
# STRUCTURAL QUALITY INDICATORS
# ============================================================

def calculate_distribution_metrics(labels):
    assigned = labels[labels != -1]
    if len(assigned) == 0:
        return {
            "discovered_groups": 0,
            "assigned_records": 0,
            "unassigned_records": int(len(labels)),
            "coverage_percentage": 0.0,
            "group_size_distribution": {},
            "shannon_entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_group_count": 0.0,
            "largest_group_share": 0.0,
            "herfindahl_concentration": 0.0,
        }

    counts = Counter(int(x) for x in assigned)
    values = np.array(list(counts.values()), dtype=float)
    probabilities = values / values.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    group_count = len(counts)
    normalized_entropy = (
        entropy / math.log(group_count) if group_count > 1 else 0.0
    )

    return {
        "discovered_groups": int(group_count),
        "assigned_records": int(len(assigned)),
        "unassigned_records": int(len(labels) - len(assigned)),
        "coverage_percentage": round(len(assigned) / len(labels) * 100, 2),
        "group_size_distribution": {
            str(k): int(v) for k, v in sorted(counts.items())
        },
        "shannon_entropy": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "effective_group_count": round(float(np.exp(entropy)), 2),
        "largest_group_share": round(float(probabilities.max()), 4),
        "herfindahl_concentration": round(
            float(np.sum(probabilities ** 2)), 4
        ),
    }


def calculate_silhouette(embeddings, labels):
    assigned_mask = labels != -1
    assigned_labels = labels[assigned_mask]
    assigned_embeddings = embeddings[assigned_mask]

    unique = np.unique(assigned_labels)
    if len(unique) < 2 or len(assigned_labels) <= len(unique):
        return None

    try:
        score = silhouette_score(
            assigned_embeddings,
            assigned_labels,
            metric="cosine",
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(assigned_labels)),
            random_state=RANDOM_STATE,
        )
        return round(float(score), 4)
    except Exception:
        return None


def build_summary(
    original_count,
    valid_count,
    text_column,
    labels,
    cluster_details,
    embeddings,
    dataset_path,
):
    distribution = calculate_distribution_metrics(labels)
    cohesion_values = [
        item["mean_cosine_cohesion"]
        for item in cluster_details.values()
    ]
    mean_cohesion = (
        float(np.mean(cohesion_values)) if cohesion_values else None
    )

    summary = {
        "metric": "Metric 05 - Unsupervised Semantic Domain Discovery",
        "version": FROZEN_VERSION,
        "method_id": FROZEN_METHOD_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "dataset_sha256": sha256_file(dataset_path),
            "environment": environment_metadata(),
            "random_state": RANDOM_STATE,
            "embedding_batch_size": BATCH_SIZE,
            "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
            "hdbscan_parameters": {
                "min_cluster_size_rule": "clip(round(sqrt(n_records) * 0.5), 10, 50)",
                "min_samples_rule": "max(2, round(min_cluster_size * 0.25))",
                "metric": "euclidean",
                "cluster_selection_method": "eom",
            },
            "umap_parameters": {
                "n_neighbors_rule": "min(15, max(2, n_records - 1))",
                "n_components_rule": "min(5, embedding_dimensions, n_records - 2)",
                "min_dist": 0.0,
                "metric": "cosine",
                "random_state": RANDOM_STATE,
                "transform_seed": RANDOM_STATE,
                "n_jobs": 1,
                "compatibility_layout": True,
            },
        },
        "method": {
            "embedding_model": MODEL_NAME,
            "clustering_algorithm": "HDBSCAN",
            "dimensionality_reduction": "UMAP when available",
            "group_description_method": "Class-based TF-IDF with Telugu-aware tokenization",
            "group_description_status": "Automatic keyword descriptors; not human-validated labels",
            "silhouette_evaluation_space": "Original normalized multilingual embedding space (cosine distance)",
            "fixed_domain_taxonomy_used": False,
            "manual_labels_required": False,
        },
        "dataset": {
            "total_records": int(original_count),
            "evaluated_records": int(valid_count),
            "empty_records": int(original_count - valid_count),
            "text_column": str(text_column),
        },
        "discovery": distribution,
        "cluster_structure": {
            "mean_group_cohesion": (
                round(mean_cohesion, 4) if mean_cohesion is not None else None
            ),
            "silhouette_score_cosine": calculate_silhouette(
                embeddings, labels
            ),
            "groups": {
                str(k): v for k, v in cluster_details.items()
            },
        },
        "interpretation": {
            "domain_diversity": (
                "Described by the number of discovered semantic groups "
                "and effective group count. More groups are not automatically "
                "better; fragmentation and dataset size matter."
            ),
            "domain_balance": (
                "Described by normalized entropy across discovered groups."
            ),
            "domain_concentration": (
                "Described by largest-group share and Herfindahl concentration."
            ),
            "discovery_reliability": (
                "Described using group cohesion, silhouette and unassigned "
                "record coverage. These are unsupervised structural indicators, "
                "not classification accuracy."
            ),
        },
        "limitations": [
            "Discovered groups describe recurring semantic structure, not necessarily universal subject domains.",
            "Automatically generated group descriptions are keyword summaries, not verified human labels.",
            "Multilingual embeddings may perform unevenly across languages, dialects, transliteration and code-mixing.",
            "Clustering results depend on embedding model and clustering parameters.",
            "Unassigned records may be outliers, short texts, mixed-topic records or records not well represented by the embedding space.",
            "Domain diversity does not establish representativeness of an intended population.",
            "No single universal numeric quality grade is assigned by this version.",
        ],
    }
    return summary


# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(all_records, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records.to_csv(RECORDS_OUTPUT, index=False, encoding="utf-8-sig")

    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\nOutputs saved:")
    print(f"Record-level results: {RECORDS_OUTPUT}")
    print(f"Summary report: {SUMMARY_OUTPUT}")


def print_report(summary):
    discovery = summary["discovery"]
    structure = summary["cluster_structure"]

    print("\n" + "=" * 64)
    print("METRIC 05: UNSUPERVISED SEMANTIC DOMAIN DISCOVERY")
    print("=" * 64)
    print(f"Total records: {summary['dataset']['total_records']}")
    print(f"Evaluated records: {summary['dataset']['evaluated_records']}")
    print(f"Discovered semantic groups: {discovery['discovered_groups']}")
    print(f"Assigned records: {discovery['assigned_records']}")
    print(f"Unassigned records: {discovery['unassigned_records']}")
    print(f"Discovery coverage: {discovery['coverage_percentage']}%")
    print(f"Effective group count: {discovery['effective_group_count']}")
    print(f"Normalized entropy: {discovery['normalized_entropy']}")
    print(f"Largest group share: {discovery['largest_group_share']}")
    print(f"Mean group cohesion: {structure['mean_group_cohesion']}")
    print(f"Cosine silhouette: {structure['silhouette_score_cosine']}")

    print("\nDISCOVERED GROUPS")
    for group_id, details in structure["groups"].items():
        print(
            f"\nGroup {group_id} | Records: {details['record_count']} "
            f"| Cohesion: {details['mean_cosine_cohesion']}"
        )
        print(f"Description: {details['generated_description']}")
        print(f"Top terms: {', '.join(details['top_terms'])}")

    print("\nNo universal quality grade is assigned.")
    print("=" * 64)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised semantic domain discovery for text datasets."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_DATA_PATH),
        help="Path to a CSV or Excel dataset.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Name of the column containing text.",
    )
    args = parser.parse_args()

    data_path = Path(args.dataset).expanduser()

    print("=" * 64)
    print("TELCO-QA - METRIC 05 V2.1 FROZEN")
    print("Unsupervised Semantic Domain Discovery")
    print("=" * 64)

    df = load_dataset(data_path)
    text_column = identify_text_column(df, args.text_column)
    all_records, valid_records = prepare_records(df, text_column)

    print("\nLoading multilingual embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = valid_records["processed_text"].tolist()
    embeddings = generate_embeddings(texts, model)

    labels, strengths = discover_clusters(embeddings)
    descriptions, cluster_details = describe_clusters(
        texts, embeddings, labels
    )

    classified = valid_records.copy()
    classified["semantic_group_id"] = labels
    classified["cluster_strength"] = (
        strengths if strengths is not None else np.zeros(len(labels))
    )
    classified["discovered_group_description"] = [
        descriptions.get(int(label), "Unassigned / Noise")
        if label != -1 else "Unassigned / Noise"
        for label in labels
    ]

    summary = build_summary(
        original_count=len(all_records),
        valid_count=len(valid_records),
        text_column=text_column,
        labels=labels,
        cluster_details=cluster_details,
        embeddings=embeddings,
        dataset_path=data_path,
    )

    # Merge analysis columns back into every original row.
    output_records = all_records.copy()
    output_records["semantic_group_id"] = pd.NA
    output_records["cluster_strength"] = 0.0
    output_records["discovered_group_description"] = "Empty / No Text"

    output_records.loc[
        classified.index, "semantic_group_id"
    ] = classified["semantic_group_id"]
    output_records.loc[
        classified.index, "cluster_strength"
    ] = classified["cluster_strength"]
    output_records.loc[
        classified.index, "discovered_group_description"
    ] = classified["discovered_group_description"]

    save_outputs(output_records, summary)
    print_report(summary)


if __name__ == "__main__":
    main()
