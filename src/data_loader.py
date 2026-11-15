
from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"


def load_dataset():
    """
    Load the Telugu-English dataset from the Excel file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    df = pd.read_excel(DATASET_PATH)

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if "text" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'text' column."
        )

    return df


if __name__ == "__main__":
    dataset = load_dataset()

    print("\nDataset loaded successfully!")
    print("-" * 40)

    print(f"Total records: {len(dataset)}")
    print(f"Total columns: {len(dataset.columns)}")

    print("\nColumn names:")
    print(dataset.columns.tolist())

    print("\nFirst 5 records:")
    print(dataset.head())
