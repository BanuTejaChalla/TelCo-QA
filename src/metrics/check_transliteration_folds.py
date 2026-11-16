
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_grouped_dataset.csv"
)


def evaluate_folds(df, n_splits):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    X = df.drop(columns=["label"])
    y = df["label"]
    groups = df["group_id"]

    print(f"\n{'=' * 55}")
    print(f"STRATIFIED GROUP {n_splits}-FOLD EVALUATION")
    print(f"{'=' * 55}")

    all_valid = True

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_groups = set(train_df["group_id"])
        test_groups = set(test_df["group_id"])

        overlap = train_groups.intersection(test_groups)

        train_classes = set(train_df["label"])
        test_classes = set(test_df["label"])

        valid = (
            len(overlap) == 0
            and len(train_classes) == 2
            and len(test_classes) == 2
        )

        if not valid:
            all_valid = False

        print(f"\nFold {fold}")
        print("-" * 30)

        print(f"Training pairs : {len(train_df)}")
        print(f"Testing pairs  : {len(test_df)}")

        print(f"Training groups: {len(train_groups)}")
        print(f"Testing groups : {len(test_groups)}")

        print(
            f"Training labels: "
            f"{train_df['label'].value_counts().to_dict()}"
        )

        print(
            f"Testing labels : "
            f"{test_df['label'].value_counts().to_dict()}"
        )

        print(f"Group overlap  : {len(overlap)}")
        print(f"Fold valid     : {valid}")

    print(f"\nOverall feasibility: {all_valid}")

    return all_valid


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Grouped dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nTRANSLITERATION FOLD FEASIBILITY")
    print("--------------------------------")

    print(f"Total pairs : {len(df)}")
    print(f"Total groups: {df['group_id'].nunique()}")

    for n_splits in [5, 4, 3]:
        evaluate_folds(df, n_splits)


if __name__ == "__main__":
    main()
