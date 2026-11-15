
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transliteration_feature_dataset.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    # Separate the two annotation classes.
    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]

    # Select numerical features only.
    excluded_columns = {
        "label",
        "frequency_1",
        "frequency_2",
    }

    feature_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]

    # Calculate class-wise feature averages.
    comparison = pd.DataFrame({
        "Positive (1)": positive[feature_columns].mean(),
        "Negative (0)": negative[feature_columns].mean(),
    })

    comparison["Absolute Difference"] = (
        comparison["Positive (1)"]
        - comparison["Negative (0)"]
    ).abs()

    comparison = comparison.sort_values(
        by="Absolute Difference",
        ascending=False
    )

    print("\nTRANSLITERATION FEATURE ANALYSIS")
    print("--------------------------------")

    print(f"Total pairs       : {len(df)}")
    print(f"Positive pairs    : {len(positive)}")
    print(f"Negative pairs    : {len(negative)}")
    print(f"Numerical features : {len(feature_columns)}")

    print("\nCLASS-WISE FEATURE COMPARISON")
    print("-----------------------------")

    print(comparison.round(3).to_string())

    # Show the range of values for each class.
    print("\nFEATURE RANGE COMPARISON")
    print("------------------------")

    for feature in feature_columns:
        positive_min = positive[feature].min()
        positive_max = positive[feature].max()

        negative_min = negative[feature].min()
        negative_max = negative[feature].max()

        print(f"\n{feature}")
        print(
            f"  Positive: {positive_min:.3f} to "
            f"{positive_max:.3f}"
        )
        print(
            f"  Negative: {negative_min:.3f} to "
            f"{negative_max:.3f}"
        )

    # Save the comparison for future reference.
    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "transliteration_feature_comparison.csv"
    )

    comparison.to_csv(output_file)

    print(f"\nComparison saved to:\n{output_file}")


if __name__ == "__main__":
    main()
