
# =====================================================================
# METRIC 05 — UNIVERSAL DOMAIN REPRESENTATIVENESS
# Yama THX-01 | Labelled and Unlabelled Datasets
#
# LABELLED:
#   Original labels = primary Metric 05 distribution
#   Yama = independent classification and comparison
#
# UNLABELLED:
#   Yama predictions = primary Metric 05 distribution
#
# Inputs:
#   1. Dataset upload
#   2. Existing domain-label column (or blank)
#   3. Preferred target domain
#
# No clustering, embeddings, or manual review.
# =====================================================================

import os
import re
import json
import time
import hashlib
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import thx01

from tqdm.auto import tqdm
from google.colab import files

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

MODEL_NAME = "LatentAILabs/yama-indic"

CHECKPOINT_EVERY = 320  # 5 groups of 64 records
OUTPUT_BASE = Path("/content/metric05_results")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Your tested Yama taxonomy, with General added.
YAMA_DOMAINS = {
    "telecommunications": (
        "Mobile networks, telecom operators, SIM cards, recharge, "
        "mobile data, internet connectivity, or telecom customer service."
    ),
    "technology": (
        "Technology, software, computers, apps, gadgets, or digital tools "
        "not primarily about telecom service."
    ),
    "entertainment": (
        "Movies, music, television, celebrities, or entertainment."
    ),
    "education": (
        "Schools, colleges, exams, learning, or academic matters."
    ),
    "healthcare": (
        "Health, illness, hospitals, doctors, or medical care."
    ),
    "finance_business": (
        "Money, banking, business, jobs, or commerce."
    ),
    "sports": (
        "Sports, games, teams, or athletes."
    ),
    "food": (
        "Food, cooking, restaurants, or eating."
    ),
    "travel": (
        "Travel, transport, tourism, or places to visit."
    ),
    "social_issues": (
        "Social, family, community, or public issues."
    ),
    "general": (
        "General everyday conversations, greetings, casual discussions, "
        "personal expressions, common statements, and miscellaneous "
        "topics that do not clearly belong to another specific domain."
    ),
    "other": (
        "Text that is unclear, unrelated, or does not fit any of the "
        "available domain categories."
    ),
}

# =====================================================================
# 2. LOAD THE YAMA MODEL
# =====================================================================

print("=" * 70)
print("METRIC 05 — DOMAIN REPRESENTATIVENESS")
print("=" * 70)

if "agent" not in globals():
    print(f"\nLoading Yama model: {MODEL_NAME}")
    model_start = time.time()
    agent = thx01.load(MODEL_NAME)
    print(f"Model loaded in {time.time() - model_start:.1f} seconds.")
else:
    print("\nUsing the already-loaded Yama model.")

if not hasattr(agent, "system_one"):
    raise AttributeError(
        "The loaded model does not have system_one(). "
        "Load Yama using your tested loading method."
    )

# =====================================================================
# 3. UPLOAD DATASET
# =====================================================================

print("\nUpload your Excel or CSV dataset.")

uploaded = files.upload()

if not uploaded:
    raise ValueError("No dataset uploaded.")

filename = next(iter(uploaded.keys()))
extension = Path(filename).suffix.lower()

if extension in [".xlsx", ".xls"]:
    df = pd.read_excel(filename)
elif extension == ".csv":
    df = pd.read_csv(filename, encoding="utf-8-sig")
else:
    raise ValueError("Only Excel and CSV files are supported.")

df = df.reset_index(drop=True)

if df.empty:
    raise ValueError("The dataset is empty.")

print(f"\nDataset: {filename}")
print(f"Records: {len(df):,}")
print("\nAvailable columns:")

for column in df.columns:
    print("-", column)

# =====================================================================
# 4. AUTOMATIC TEXT COLUMN DETECTION
# =====================================================================

text_candidates = [
    "text", "content", "sentence", "utterance",
    "message", "post", "comment", "tweet",
    "review", "query", "description"
]

column_lookup = {
    str(c).strip().lower(): c for c in df.columns
}

text_column = None

for candidate in text_candidates:
    if candidate in column_lookup:
        text_column = column_lookup[candidate]
        break

if text_column is None:

    excluded_columns = {
        "id", "sentiment", "source", "topic",
        "label", "category", "domain",
        "language", "language_code", "language_codes",
        "word_count", "language_code_count", "count_match"
    }

    candidates = [
        c for c in df.select_dtypes(include=["object", "string"]).columns
        if str(c).strip().lower() not in excluded_columns
    ]

    if not candidates:
        raise ValueError(
            "Text column could not be detected. "
            "Rename the text column to 'text' and rerun."
        )

    text_column = max(
        candidates,
        key=lambda c: (
            df[c].dropna().astype(str).str.len().mean()
            if df[c].notna().any() else 0
        )
    )

print(f"\nAutomatically detected text column: {text_column}")

# =====================================================================
# 5. IDENTIFY WHETHER DOMAIN LABELS EXIST
# =====================================================================

print("\nEnter the column containing existing DOMAIN labels.")
print("If the dataset has no domain labels, press Enter.")

label_column_input = input(
    "\nDomain-label column [blank if none]: "
).strip()

if label_column_input:

    if label_column_input not in df.columns:
        raise ValueError(
            f"Column '{label_column_input}' was not found."
        )

    if label_column_input == text_column:
        raise ValueError("Text column cannot also be the domain column.")

    label_column = label_column_input

    original_labels = (
        df[label_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    original_labels = original_labels[
        original_labels != ""
    ]

    if original_labels.empty:
        raise ValueError("The selected domain column contains no labels.")

    existing_domains = sorted(original_labels.unique().tolist())

    if len(existing_domains) < 2:
        raise ValueError(
            "At least two different domain categories are needed "
            "for domain diversity analysis."
        )

    print("\nLABELLED DATASET DETECTED.")
    print(f"Existing domain categories: {len(existing_domains)}")

    for i, domain in enumerate(existing_domains, 1):
        print(f"{i}. {domain}")

    # The existing dataset labels determine the primary taxonomy.
    # Ask for the preferred target from those original labels.

    target_input = input(
        "\nEnter preferred target domain from the above list: "
    ).strip()

    matching_targets = [
        d for d in existing_domains
        if d.lower() == target_input.lower()
    ]

    if not matching_targets:
        raise ValueError(
            "Target domain must match one of the existing dataset labels."
        )

    target_domain = matching_targets[0]
    labelled_mode = True

else:

    label_column = None
    existing_domains = []
    target_domain = None
    labelled_mode = False

    print("\nUNLABELLED DATASET DETECTED.")
    print("Yama will assign domain labels directly.")

# =====================================================================
# 6. CREATE YAMA QUESTION
# =====================================================================

def make_domain_key(label, used_keys):

    key = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        str(label).strip().lower()
    ).strip("_")

    if not key:
        key = "domain"

    original_key = key
    counter = 2

    while key in used_keys:
        key = f"{original_key}_{counter}"
        counter += 1

    used_keys.add(key)
    return key


if labelled_mode:

    # Yama uses the actual existing domain categories.
    # This enables a direct prediction-to-label comparison.

    yama_key_to_label = {}
    criteria = {}
    used_keys = set()

    for label in existing_domains:

        key = make_domain_key(label, used_keys)

        yama_key_to_label[key] = label

        criteria[key] = (
            f"The main subject or topic is {label}. "
            f"Choose this category when the Telugu-English text "
            f"primarily discusses or relates to {label}."
        )

    # Add an outside-category option if the original taxonomy
    # does not already contain an Other category.

    has_other = any(
        str(label).strip().lower() == "other"
        for label in existing_domains
    )

    if not has_other:
        other_key = make_domain_key("other", used_keys)
        yama_key_to_label[other_key] = None
        criteria[other_key] = (
            "The text does not belong to any of the supplied "
            "existing domain categories."
        )

    domain_question = {
        "domain": {
            "type": "choice",
            "instructions": (
                "What is this Telugu-English code-mixed text mainly about? "
                "Choose the best matching category from the provided "
                "domain labels. Use the outside-category option only "
                "when none of the provided categories fits."
            ),
            "criteria": criteria,
        }
    }

else:

    # Unlabelled datasets use the fixed, tested Yama taxonomy.

    yama_key_to_label = {
        key: key for key in YAMA_DOMAINS
    }

    domain_question = {
        "domain": {
            "type": "choice",
            "instructions": (
                "What is this Telugu-English code-mixed text mainly about?"
            ),
            "criteria": YAMA_DOMAINS,
        }
    }

# =====================================================================
# 7. CREATE A SEPARATE OUTPUT FOLDER FOR THIS DATASET
# =====================================================================

text_hash = hashlib.sha256(
    pd.util.hash_pandas_object(
        df[text_column].fillna("").astype(str),
        index=False
    ).values.tobytes()
).hexdigest()[:12]

label_hash = "unlabelled"

if labelled_mode:
    label_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            df[label_column].fillna("").astype(str),
            index=False
        ).values.tobytes()
    ).hexdigest()[:12]

safe_name = re.sub(
    r"[^A-Za-z0-9_-]+",
    "_",
    Path(filename).stem
)

signature = hashlib.sha256(
    (
        f"{filename}|{len(df)}|{text_hash}|"
        f"{label_hash}|{label_column}|{MODEL_NAME}"
    ).encode()
).hexdigest()[:12]

output_dir = OUTPUT_BASE / f"{safe_name}_{signature}"
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = output_dir / "metric05_checkpoint.csv"

print(f"\nOutput folder: {output_dir}")

# =====================================================================
# 8. INITIALIZE RESULT DATAFRAME
# =====================================================================

result_df = df.copy()
result_df.insert(0, "row_id", np.arange(len(df)))

output_columns = [
    "yama_domain",
    "yama_confidence",
    "yama_probabilities",
    "yama_raw_response",
    "prediction_status",
]

for col in output_columns:
    result_df[col] = None

# Resume only when the checkpoint belongs to this dataset signature.

metadata_path = output_dir / "run_metadata.json"

if metadata_path.exists() and checkpoint_path.exists():

    with open(metadata_path, "r", encoding="utf-8") as f:
        saved_metadata = json.load(f)

    if saved_metadata.get("signature") == signature:

        print("\nMatching checkpoint found. Recovering completed records.")

        old = pd.read_csv(checkpoint_path)

        old = old.drop_duplicates(
            subset=["row_id"],
            keep="last"
        )

        for _, row in old.iterrows():

            rid = int(row["row_id"])

            if 0 <= rid < len(result_df):

                for col in output_columns:
                    if col in old.columns:
                        result_df.at[rid, col] = row[col]

# Save metadata.
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "signature": signature,
            "filename": filename,
            "records": len(df),
            "text_column": str(text_column),
            "label_column": str(label_column),
            "model": MODEL_NAME,
            "labelled_mode": labelled_mode,
        },
        f,
        indent=4
    )

completed_statuses = {"success", "empty_text"}

completed_ids = set(
    result_df.index[
        result_df["prediction_status"].isin(completed_statuses)
    ].tolist()
)

pending_ids = [
    i for i in range(len(df))
    if i not in completed_ids
]

print(f"Completed records recovered: {len(completed_ids):,}")
print(f"Records remaining: {len(pending_ids):,}")

# =====================================================================
# 9. YAMA CLASSIFICATION — EXACT SYSTEM_ONE METHOD
# =====================================================================

def classify_text(text):

    response = agent.system_one(
        text,
        domain_question
    )

    answer = response["answers"]["domain"]

    return {
        "domain": answer.get("choice"),
        "confidence": answer.get("confidence"),
        "probabilities": answer.get("probabilities", {}),
        "raw_response": response,
    }


print("\nStarting Yama classification...")

start_time = time.time()

for batch_start in range(0, len(pending_ids), 64):

    batch_ids = pending_ids[batch_start:batch_start + 64]

    batch_number = batch_start // 64 + 1

    for rid in tqdm(
        batch_ids,
        desc=f"Processing group {batch_number}"
    ):

        text_value = df.at[rid, text_column]

        if pd.isna(text_value) or not str(text_value).strip():

            result_df.at[rid, "prediction_status"] = "empty_text"
            continue

        try:

            prediction = classify_text(str(text_value))

            result_df.at[rid, "yama_domain"] = prediction["domain"]

            result_df.at[rid, "yama_confidence"] = prediction["confidence"]

            result_df.at[rid, "yama_probabilities"] = json.dumps(
                prediction["probabilities"],
                ensure_ascii=False
            )

            result_df.at[rid, "yama_raw_response"] = json.dumps(
                prediction["raw_response"],
                ensure_ascii=False,
                default=str
            )

            result_df.at[rid, "prediction_status"] = (
                "success"
                if prediction["domain"] in yama_key_to_label
                else "parse_error"
            )

        except Exception as exc:

            result_df.at[rid, "prediction_status"] = "error"

            result_df.at[rid, "yama_raw_response"] = (
                f"{type(exc).__name__}: {exc}"
            )

    # Checkpoint after every 320 attempted records.

    records_processed = batch_start + len(batch_ids)

    if (
        records_processed % CHECKPOINT_EVERY == 0
        or records_processed == len(pending_ids)
    ):

        result_df.to_csv(
            checkpoint_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"\nCheckpoint saved: "
            f"{records_processed:,}/{len(pending_ids):,} "
            f"pending records processed."
        )

# Final checkpoint.
result_df.to_csv(
    checkpoint_path,
    index=False,
    encoding="utf-8-sig"
)

elapsed_time = time.time() - start_time

# =====================================================================
# 10. PREPARE SUCCESSFUL PREDICTIONS
# =====================================================================

successful_df = result_df[
    result_df["prediction_status"] == "success"
].copy()

successful_df["yama_confidence"] = pd.to_numeric(
    successful_df["yama_confidence"],
    errors="coerce"
)

if successful_df.empty:
    raise RuntimeError(
        "No successful Yama predictions. Check the error records."
    )

# Convert Yama choice keys to readable/original domain labels.
successful_df["yama_domain_label"] = (
    successful_df["yama_domain"].map(yama_key_to_label)
)

# =====================================================================
# 11. SELECT PRIMARY DISTRIBUTION
# =====================================================================

if labelled_mode:

    # PRIMARY SOURCE = ORIGINAL DATASET LABELS.
    # Do not replace these with Yama predictions.

    primary_df = result_df[
        result_df[label_column].notna()
    ].copy()

    primary_df["primary_domain"] = (
        primary_df[label_column].astype(str).str.strip()
    )

    primary_df = primary_df[
        primary_df["primary_domain"] != ""
    ]

    primary_source = "Original dataset domain labels"

    primary_target = target_domain

else:

    # PRIMARY SOURCE = YAMA-ASSIGNED LABELS.

    primary_df = successful_df[
        successful_df["yama_domain_label"].notna()
    ].copy()

    primary_df["primary_domain"] = (
        primary_df["yama_domain_label"].astype(str)
    )

    primary_source = "Yama-assigned domain labels"

    # Show only categories actually assigned by Yama.
    discovered_domains = sorted(
        primary_df["primary_domain"].unique().tolist()
    )

    print("\n" + "=" * 65)
    print("DOMAINS IDENTIFIED BY YAMA")
    print("=" * 65)

    for i, domain in enumerate(discovered_domains, 1):
        print(f"{i}. {domain}")

    if not discovered_domains:
        raise RuntimeError("Yama did not assign any usable domains.")

    target_input = input(
        "\nEnter your preferred target domain from this list: "
    ).strip()

    matching = [
        d for d in discovered_domains
        if d.lower() == target_input.lower()
    ]

    if not matching:
        raise ValueError(
            "Please enter one of the domains displayed above."
        )

    primary_target = matching[0]

if primary_df.empty:
    raise RuntimeError("No records available for primary Metric 05 calculation.")

# =====================================================================
# 12. CALCULATE DOMAIN DISTRIBUTION
# =====================================================================

domain_counts = primary_df["primary_domain"].value_counts()

total_primary = len(primary_df)
observed_domains = list(domain_counts.index)
k = len(observed_domains)

distribution_df = pd.DataFrame({
    "domain": observed_domains,
    "record_count": [int(domain_counts[d]) for d in observed_domains],
})

distribution_df["percentage"] = (
    distribution_df["record_count"] / total_primary * 100
)

distribution_df = distribution_df.sort_values(
    "record_count",
    ascending=False
).reset_index(drop=True)

# =====================================================================
# 13. METRIC 05 — NDDS, HHI AND DCI
# =====================================================================

proportions = (
    distribution_df["record_count"].to_numpy(dtype=float)
    / total_primary
)

nonzero = proportions[proportions > 0]

shannon_entropy = -np.sum(
    nonzero * np.log(nonzero)
)

if k > 1:
    ndds = (shannon_entropy / np.log(k)) * 100
else:
    ndds = 0.0

hhi = np.sum(proportions ** 2)
dci = hhi * 100

# =====================================================================
# 14. TARGET DOMAIN ANALYSIS
# =====================================================================

target_count = int(
    domain_counts.get(primary_target, 0)
)

target_share = (
    target_count / total_primary
) * 100

dominant_domain = distribution_df.iloc[0]["domain"]
dominant_count = int(distribution_df.iloc[0]["record_count"])

other_counts = distribution_df[
    distribution_df["domain"] != primary_target
]["record_count"]

largest_other_count = (
    int(other_counts.max()) if len(other_counts) else 0
)

records_to_become_uniquely_dominant = max(
    0,
    largest_other_count - target_count + 1
)

# =====================================================================
# 15. OPTIONAL COMPARISON WITH YAMA
# =====================================================================

agreement = None
comparison_count = 0
comparison_df = pd.DataFrame()

if labelled_mode:

    comparison_df = successful_df[
        successful_df[label_column].notna()
        & successful_df["yama_domain_label"].notna()
    ].copy()

    comparison_count = len(comparison_df)

    if comparison_count:

        original = (
            comparison_df[label_column]
            .astype(str).str.strip().str.lower()
        )

        predicted = (
            comparison_df["yama_domain_label"]
            .astype(str).str.strip().str.lower()
        )

        agreement = (original == predicted).mean() * 100

# =====================================================================
# 16. CONFIDENCE SUMMARY
# =====================================================================

mean_confidence = successful_df["yama_confidence"].mean()

# =====================================================================
# 17. SAVE OUTPUT FILES
# =====================================================================

records_path = output_dir / "metric05_records.csv"
distribution_path = output_dir / "metric05_domain_distribution.csv"
report_path = output_dir / "metric05_report.txt"
comparison_path = output_dir / "metric05_yama_comparison.csv"
zip_path = output_dir / "metric05_outputs.zip"

# Save all original columns and Yama predictions.
result_df.to_csv(
    records_path,
    index=False,
    encoding="utf-8-sig"
)

distribution_df.to_csv(
    distribution_path,
    index=False,
    encoding="utf-8-sig"
)

if labelled_mode:

    comparison_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig"
    )

# =====================================================================
# 18. GENERATE REPORT
# =====================================================================

report = [
    "=" * 70,
    "METRIC 05 — DOMAIN REPRESENTATIVENESS REPORT",
    "=" * 70,
    "",
    f"Dataset: {filename}",
    f"Total dataset records: {len(df):,}",
    f"Text column: {text_column}",
    f"Domain-label column: {label_column or 'Not available'}",
    f"Dataset type: {'Labelled' if labelled_mode else 'Unlabelled'}",
    f"Primary distribution source: {primary_source}",
    f"Preferred target domain: {primary_target}",
    f"Yama model: {MODEL_NAME}",
    f"Classification method: system_one()",
    "",
    "CLASSIFICATION SUMMARY",
    "-" * 70,
    f"Successful Yama predictions: {len(successful_df):,}",
    f"Empty text records: {(result_df['prediction_status'] == 'empty_text').sum():,}",
    f"Failed/invalid predictions: "
    f"{(~result_df['prediction_status'].isin(['success', 'empty_text'])).sum():,}",
    f"Mean Yama confidence: {mean_confidence:.4f}",
    f"Processing time: {elapsed_time:.2f} seconds",
    "",
    "PRIMARY DOMAIN DISTRIBUTION",
    "-" * 70,
]

for _, row in distribution_df.iterrows():

    report.append(
        f"{row['domain']}: "
        f"{int(row['record_count']):,} "
        f"({row['percentage']:.2f}%)"
    )

report.extend([
    "",
    "METRIC 05 RESULTS",
    "-" * 70,
    f"Number of observed domains (k): {k}",
    f"Shannon entropy: {shannon_entropy:.6f}",
    f"Normalized Domain Diversity Score (NDDS): {ndds:.2f}/100",
    f"Herfindahl-Hirschman Index (HHI): {hhi:.6f}",
    f"Domain Concentration Index (DCI): {dci:.2f}/100",
    "",
    "TARGET DOMAIN ANALYSIS",
    "-" * 70,
    f"Target domain: {primary_target}",
    f"Target records: {target_count:,}",
    f"Target domain share: {target_share:.2f}%",
    f"Most represented domain: {dominant_domain}",
    f"Most represented domain records: {dominant_count:,}",
    f"Additional target records to become uniquely dominant: "
    f"{records_to_become_uniquely_dominant:,}",
])

if labelled_mode:

    report.extend([
        "",
        "INDEPENDENT YAMA COMPARISON",
        "-" * 70,
        f"Records compared: {comparison_count:,}",
        (
            f"Exact label agreement: {agreement:.2f}%"
            if agreement is not None
            else "Exact label agreement: Not available"
        ),
        "Original dataset labels remain the primary Metric 05 labels.",
        "Yama predictions are stored separately for comparison.",
        "Agreement is not independently verified model accuracy.",
    ])

else:

    report.extend([
        "",
        "UNLABELLED DATASET METHODOLOGY",
        "-" * 70,
        "Yama directly assigned the domain labels.",
        "Yama-assigned labels are used as working labels.",
        "The target domain was selected from domains actually assigned.",
    ])

report.extend([
    "",
    "METHODOLOGICAL NOTES",
    "-" * 70,
    "NDDS measures the diversity and balance of the domain distribution.",
    "HHI and DCI measure domain concentration.",
    "These metrics do not independently establish semantic accuracy",
    "or complete statistical representativeness.",
    "Yama confidence is not equivalent to classification accuracy.",
    "",
    "=" * 70,
])

report_text = "\n".join(report)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

# =====================================================================
# 19. CREATE ZIP ARCHIVE
# =====================================================================

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as archive:

    for path in [
        records_path,
        distribution_path,
        report_path,
        checkpoint_path,
    ]:

        archive.write(path, arcname=path.name)

    if labelled_mode:
        archive.write(
            comparison_path,
            arcname=comparison_path.name
        )

# =====================================================================
# 20. DISPLAY FINAL RESULTS
# =====================================================================

print("\n" + "=" * 70)
print("METRIC 05 COMPLETED")
print("=" * 70)

print(f"\nDataset type: {'Labelled' if labelled_mode else 'Unlabelled'}")
print(f"Primary source: {primary_source}")

print("\nDOMAIN DISTRIBUTION:")
print(distribution_df.to_string(index=False))

print("\nMETRIC 05:")
print(f"NDDS: {ndds:.2f}/100")
print(f"HHI:  {hhi:.6f}")
print(f"DCI:  {dci:.2f}/100")

print("\nTARGET DOMAIN:")
print(f"Domain: {primary_target}")
print(f"Share: {target_share:.2f}%")
print(
    "Records needed to become uniquely dominant:",
    records_to_become_uniquely_dominant
)

if labelled_mode and agreement is not None:
    print(f"\nYama label agreement: {agreement:.2f}%")

print("\nOUTPUT FILES:")
print(f"Records:      {records_path}")
print(f"Distribution: {distribution_path}")
print(f"Report:       {report_path}")
print(f"Checkpoint:   {checkpoint_path}")
print(f"ZIP:          {zip_path}")

print("\n" + "=" * 70)

files.download(str(zip_path))
