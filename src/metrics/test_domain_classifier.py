
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR
    / "outputs"
    / "metric_05_domain_classifier"
    / "domain_classifier.joblib"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("\nTelCo-QA Domain Classifier")
    print("Enter a Telugu-English sentence.")
    print("Type 'exit' to quit.\n")

    while True:
        sentence = input("Sentence: ").strip()

        if sentence.lower() in {"exit", "quit"}:
            print("Testing completed.")
            break

        if not sentence:
            print("Please enter a sentence.")
            continue

        prediction = model.predict([sentence])[0]

        probabilities = model.predict_proba([sentence])[0]
        classes = model.classes_

        confidence = max(probabilities) * 100

        print(f"\nPredicted Domain: {prediction}")
        print(f"Model Confidence: {confidence:.2f}%")

        print("\nTop 3 Predictions:")

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        for domain, probability in ranked[:3]:
            print(f"  {domain}: {probability * 100:.2f}%")

        print("-" * 45)


if __name__ == "__main__":
    main()
