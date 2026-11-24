"""
Interactive smoke test for Yama domain classification.
Run from the TelCo-QA project root:
    python src/metrics/test_yama_domain.py

This tests one sentence at a time and does not read or modify the dataset.
"""

import time
import thx01


MODEL_NAME = "LatentAILabs/yama-indic"

DOMAIN_QUESTION = {
    "domain": {
        "type": "choice",
        "instructions": "What is this Telugu-English code-mixed text mainly about?",
        "criteria": {
            "telecommunications": (
                "Mobile networks, telecom operators, SIM cards, recharge, "
                "mobile data, internet connectivity, or telecom customer service."
            ),
            "technology": (
                "Technology, software, computers, apps, gadgets, or digital tools "
                "not primarily about telecom service."
            ),
            "entertainment": "Movies, music, television, celebrities, or entertainment.",
            "education": "Schools, colleges, exams, learning, or academic matters.",
            "healthcare": "Health, illness, hospitals, doctors, or medical care.",
            "finance_business": "Money, banking, business, jobs, or commerce.",
            "sports": "Sports, games, teams, or athletes.",
            "food": "Food, cooking, restaurants, or eating.",
            "travel": "Travel, transport, tourism, or places to visit.",
            "social_issues": "Social, family, community, or public issues.",
            "other": None,
        },
    }
}


def main():
    print("=" * 54)
    print("TelCo-QA | Yama Domain Classification Smoke Test")
    print("=" * 54)
    print(f"Loading model: {MODEL_NAME}")
    print("The first load may take time or download model files.")

    start = time.time()
    agent = thx01.load(MODEL_NAME)
    print(f"\nModel loaded in {time.time() - start:.1f} seconds.")
    print("Enter a code-mixed sentence. Type 'exit' to quit.\n")

    while True:
        try:
            text = input("Sentence > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTest stopped.")
            break

        if text.lower() in {"exit", "quit"}:
            print("Testing completed.")
            break

        if not text:
            print("Please enter a sentence.\n")
            continue

        try:
            result = agent.system_one(text, DOMAIN_QUESTION)
            print("\nRaw Yama response:")
            print(result)
            print("-" * 54 + "\n")
        except Exception as exc:
            print(f"\nClassification failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
