
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"
ANNOTATIONS_PATH = (
    ROOT / "data" / "processed" / "transliteration_annotations_all.csv"
)

GROUP_REPORT_PATH = (
    ROOT / "data" / "processed" / "transliteration_consistency_groups.csv"
)
SUMMARY_PATH = ROOT / "outputs" / "metric_02_consistency_score.json"


# =========================================================
# UNION-FIND
# =========================================================

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, word):
        if word not in self.parent:
            self.parent[word] = word

        if self.parent[word] != word:
            self.parent[word] = self.find(self.parent[word])

        return self.parent[word]

    def union(self, word1, word2):
        root1 = self.find(word1)
        root2 = self.find(word2)

        if root1 != root2:
            self.parent[root2] = root1


# =========================================================
# BUILD CONFIRMED EQUIVALENCE GROUPS
# =========================================================

def build_equivalence_groups(annotations):
    required = {"word_1", "word_2", "annotation"}

    if not required.issubset(annotations.columns):
        raise ValueError(
            f"Annotation file must contain columns: {required}"
        )

    uf = UnionFind()

    for _, row in annotations.iterrows():
        annotation = str(row["annotation"]).strip().upper()

        # Only manually confirmed equivalent pairs are included.
        if annotation not in {"1", "1.0", "SAME"}:
            continue

        word1 = str(row["word_1"]).strip().lower()
        word2 = str(row["word_2"]).strip().lower()

        if (
            word1
            and word2
            and word1 != word2
            and word1 != "nan"
            and word2 != "nan"
        ):
            uf.union(word1, word2)

    groups = {}

    for word in uf.parent:
        root = uf.find(word)
        groups.setdefault(root, set()).add(word)

    return [
        sorted(group)
        for group in groups.values()
        if len(group) >= 2
    ]


# =========================================================
# COUNT WORD FREQUENCIES
# =========================================================

def count_word_frequencies(dataset):
    if "text" not in dataset.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    frequencies = Counter()

    for text in dataset["text"].dropna():
        words = re.findall(r"[a-z]+", str(text).lower())
        frequencies.update(words)

    return frequencies


# =========================================================
# CALCULATE CONSISTENCY
# =========================================================

def calculate_consistency(groups, frequencies):
    results = []

    total_occurrences = 0
    dominant_occurrences = 0

    for group in groups:
        variant_counts = {
            word: frequencies[word]
            for word in group
            if frequencies[word] > 0
        }

        # A group is measurable only when at least two
        # confirmed spellings occur in the dataset.
        if len(variant_counts) < 2:
            continue

        group_total = sum(variant_counts.values())
        dominant_variant = max(
            variant_counts,
            key=variant_counts.get
        )
        dominant_count = variant_counts[dominant_variant]

        group_score = (
            dominant_count / group_total
        ) * 100

        total_occurrences += group_total
        dominant_occurrences += dominant_count

        results.append({
            "group_id": f"TG{len(results) + 1:02d}",
            "variants": ", ".join(sorted(variant_counts)),
            "variant_count": len(variant_counts),
            "total_occurrences": group_total,
            "dominant_variant": dominant_variant,
            "dominant_frequency": dominant_count,
            "consistency_score": round(group_score, 2),
            "variant_frequencies": json.dumps(
                variant_counts,
                ensure_ascii=False
            )
        })

    overall_score = (
        dominant_occurrences / total_occurrences * 100
        if total_occurrences > 0
        else 0.0
    )

    return results, overall_score, total_occurrences


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n" + "=" * 65)
    print("METRIC 02 - TRANSLITERATION CONSISTENCY SCORE")
    print("=" * 65)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Annotations not found: {ANNOTATIONS_PATH}"
        )

    dataset = pd.read_excel(DATASET_PATH)
    annotations = pd.read_csv(ANNOTATIONS_PATH)

    print(f"\nDataset records: {len(dataset)}")
    print(f"Annotation pairs: {len(annotations)}")

    print("\nAnnotation distribution:")
    print(
        annotations["annotation"]
        .astype(str)
        .str.strip()
        .value_counts(dropna=False)
        .to_string()
    )

    groups = build_equivalence_groups(annotations)

    print(f"\nConfirmed equivalent-word groups: {len(groups)}")

    frequencies = count_word_frequencies(dataset)

    results, score, total = calculate_consistency(
        groups,
        frequencies
    )

    GROUP_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = pd.DataFrame(results)
    report.to_csv(GROUP_REPORT_PATH, index=False)

    summary = {
        "metric": "Transliteration Consistency",
        "score": round(score, 2),
        "total_variant_occurrences": total,
        "confirmed_equivalence_groups": len(groups),
        "groups_with_multiple_observed_variants": len(results),
        "method": (
            "Frequency-weighted dominant-variant share "
            "across manually confirmed equivalent-word groups."
        ),
        "interpretation": (
            "Measures within-dataset spelling consistency, "
            "not linguistic correctness."
        )
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "-" * 65)
    print("GROUP-LEVEL RESULTS")
    print("-" * 65)

    if results:
        print(
            report[
                [
                    "group_id",
                    "variants",
                    "total_occurrences",
                    "dominant_variant",
                    "consistency_score"
                ]
            ].to_string(index=False)
        )
    else:
        print(
            "No confirmed groups with multiple observed variants."
        )

    print("\n" + "=" * 65)
    print(f"OVERALL CONSISTENCY SCORE: {score:.2f}%")
    print("=" * 65)

    print(f"\nGroup report saved: {GROUP_REPORT_PATH}")
    print(f"Summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
