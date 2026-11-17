
from pathlib import Path
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_feature_dataset_reviewed.csv"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_grouped_dataset_reviewed.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_group_summary_reviewed.csv"
)


class UnionFind:
    """Track connected word families using disjoint sets."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, word):
        if word not in self.parent:
            self.parent[word] = word
            self.rank[word] = 0

    def find(self, word):
        if self.parent[word] != word:
            self.parent[word] = self.find(self.parent[word])

        return self.parent[word]

    def union(self, word_1, word_2):
        root_1 = self.find(word_1)
        root_2 = self.find(word_2)

        if root_1 == root_2:
            return

        if self.rank[root_1] < self.rank[root_2]:
            root_1, root_2 = root_2, root_1

        self.parent[root_2] = root_1

        if self.rank[root_1] == self.rank[root_2]:
            self.rank[root_1] += 1


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"word_1", "word_2", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df[["word_1", "word_2", "label"]].isna().any().any():
        raise ValueError(
            "Word pairs or labels contain missing values."
        )

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Labels must contain only 0 and 1."
        )

    # Build connected word families.
    union_find = UnionFind()

    for _, row in df.iterrows():
        word_1 = str(row["word_1"]).strip().lower()
        word_2 = str(row["word_2"]).strip().lower()

        if not word_1 or not word_2:
            raise ValueError("Empty word found in dataset.")

        union_find.add(word_1)
        union_find.add(word_2)
        union_find.union(word_1, word_2)

    # Collect words belonging to each connected component.
    components = defaultdict(list)

    for word in union_find.parent:
        root = union_find.find(word)
        components[root].append(word)

    # Sort components to create reproducible group IDs.
    sorted_components = sorted(
        [sorted(words) for words in components.values()],
        key=lambda words: words
    )

    word_to_group = {}

    for group_number, words in enumerate(sorted_components, start=1):
        group_id = f"G{group_number:02d}"

        for word in words:
            word_to_group[word] = group_id

    # Assign a group ID to every annotated pair.
    result_df = df.copy()

    result_df["group_id"] = result_df.apply(
        lambda row: word_to_group[
            str(row["word_1"]).strip().lower()
        ],
        axis=1
    )

    # Verify that both words in every pair belong to the same group.
    for _, row in result_df.iterrows():
        group_1 = word_to_group[
            str(row["word_1"]).strip().lower()
        ]

        group_2 = word_to_group[
            str(row["word_2"]).strip().lower()
        ]

        if group_1 != group_2:
            raise AssertionError(
                "A word pair was assigned to different groups."
            )

    # Create a group-level summary.
    summary_rows = []

    for group_id, group_df in result_df.groupby("group_id"):
        words = sorted(
            set(group_df["word_1"].str.lower())
            | set(group_df["word_2"].str.lower())
        )

        positive_count = int((group_df["label"] == 1).sum())
        negative_count = int((group_df["label"] == 0).sum())

        summary_rows.append({
            "group_id": group_id,
            "number_of_words": len(words),
            "number_of_pairs": len(group_df),
            "positive_pairs": positive_count,
            "negative_pairs": negative_count,
            "class_distribution": (
                "Mixed"
                if positive_count > 0 and negative_count > 0
                else "Positive only"
                if positive_count > 0
                else "Negative only"
            ),
            "words": ", ".join(words),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save both outputs.
    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(OUTPUT_DATASET, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    # Display results.
    print("\nTRANSLITERATION GROUP ANALYSIS")
    print("--------------------------------")

    print(f"Total annotated pairs : {len(result_df)}")
    print(f"Unique words          : {len(union_find.parent)}")
    print(f"Connected groups      : {len(summary_df)}")

    print("\nGROUP SIZE DISTRIBUTION")
    print("-----------------------")

    print(
        summary_df["number_of_pairs"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCLASS DISTRIBUTION ACROSS GROUPS")
    print("--------------------------------")

    print(
        summary_df["class_distribution"]
        .value_counts()
        .to_string()
    )

    print("\nGROUP SUMMARY")
    print("-------------")

    print(
        summary_df[
            [
                "group_id",
                "number_of_words",
                "number_of_pairs",
                "positive_pairs",
                "negative_pairs",
                "class_distribution",
            ]
        ].to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("------------")
    print(f"Grouped dataset : {OUTPUT_DATASET}")
    print(f"Group summary   : {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
