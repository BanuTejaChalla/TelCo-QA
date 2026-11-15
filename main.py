
from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset


def main():
    print("\n" + "=" * 55)
    print("          TelCo-QA | Dataset Quality Framework")
    print("=" * 55)

    # Step 1: Load dataset
    print("\n[1] Loading dataset...")

    df = load_dataset()

    print("    Dataset loaded successfully!")
    print(f"    Total records : {len(df)}")
    print(f"    Total columns : {len(df.columns)}")

    # Step 2: Preprocess dataset
    print("\n[2] Preprocessing dataset...")

    processed_df = preprocess_dataset(df)

    print("    Preprocessing completed!")

    # Step 3: Display preprocessing statistics
    print("\n" + "-" * 55)
    print("              PREPROCESSING SUMMARY")
    print("-" * 55)

    print(f"Original records  : {len(df)}")
    print(f"Processed records : {len(processed_df)}")
    print(f"Empty records     : {processed_df['is_empty'].sum()}")

    # Step 4: Display text preprocessing preview
    print("\n" + "-" * 55)
    print("              TEXT PREPROCESSING PREVIEW")
    print("-" * 55)

    for index, row in processed_df.head(5).iterrows():

        print(f"\nRecord {index + 1}")
        print("-" * 40)

        print(f"Original   : {row['original_text']}")
        print(f"Cleaned    : {row['cleaned_text']}")
        print(f"Normalized : {row['normalized_text']}")
        print(f"Empty      : {row['is_empty']}")

    # Step 5: Completion message
    print("\n" + "=" * 55)
    print("       DATASET PREPROCESSING COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
