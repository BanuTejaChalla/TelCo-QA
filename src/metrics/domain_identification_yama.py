"""
Standalone Yama domain identification for TelCo-QA.
Run from repository root: python src/domain_identification.py
Does not modify Metric 5 V2.1 or the original dataset.
Install: pip install thx01 pandas openpyxl
"""
from pathlib import Path
import json
import time
import pandas as pd
import thx01

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "domain_identification_yama.csv"
MODEL_ID = "LatentAILabs/yama-indic"
BATCH_SIZE = 32

QUESTION = {
    "type": "choice",
    "instructions": (
        "Which broad semantic domain is this Telugu, Romanized Telugu, or "
        "Telugu-English code-mixed text mainly about? Choose one best domain "
        "based on meaning. If none fits, choose Other / General."
    ),
    "criteria": {
        "telecommunications": "Mobile networks, telecom operators, SIM cards, recharge, data plans, call quality, internet connectivity, broadband, network coverage, telecom customer service.",
        "technology": "Software, apps, computers, programming, gadgets, digital products, and technology issues not primarily about telecom.",
        "entertainment_media": "Movies, television, music, celebrities, streaming, social media creators, and entertainment.",
        "education": "Schools, colleges, universities, exams, courses, learning, teachers, students, and academic matters.",
        "healthcare": "Health, illness, medical treatment, hospitals, doctors, medicine, and wellbeing.",
        "politics_government": "Political parties, elections, politicians, government policy, and public administration.",
        "finance_business": "Banking, money, investments, commerce, jobs, employment, business, and economic matters.",
        "sports": "Sports events, teams, athletes, matches, and sporting performance.",
        "food": "Food, cooking, restaurants, meals, recipes, and dining.",
        "travel_places": "Tourism, trips, transport journeys, destinations, and travel experiences.",
        "social_issues": "Relationships, family, personal experiences, social concerns, community matters, and everyday life not fitting another domain.",
        "other_general": None,
    },
}

def to_plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return to_plain(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return to_plain(vars(value))
    return str(value)

def locate_dataset():
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"

    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_data_dir}"
        )

    dataset_files = [
        file
        for file in raw_data_dir.iterdir()
        if file.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and "telco" in file.name.lower()
    ]

    if len(dataset_files) == 1:
        return dataset_files[0]

    if not dataset_files:
        raise FileNotFoundError(
            f"No TelCo-QA dataset found in {raw_data_dir}. "
            "Check the dataset filename."
        )

    raise RuntimeError(
        f"Multiple possible datasets found in {raw_data_dir}:\n"
        + "\n".join(str(file) for file in dataset_files)
    )

def parse_choice(result):
    obj = to_plain(result)
    if isinstance(obj, dict) and isinstance(obj.get("answers"), dict):
        obj = obj["answers"].get("domain", obj)
    elif isinstance(obj, dict) and "domain" in obj:
        obj = obj["domain"]

    if not isinstance(obj, dict):
        return "PARSE_REVIEW", None, "", None

    label = None
    for key in ("choice", "label", "answer", "selected", "prediction", "value"):
        v = obj.get(key)
        if isinstance(v, str) and v in QUESTION["criteria"]:
            label = v
            break

    confidence = None
    runner_label = ""
    runner_conf = None
    for key in ("probabilities", "probs", "distribution", "scores"):
        values = obj.get(key)
        if isinstance(values, dict):
            pairs = []
            for k, v in values.items():
                try:
                    pairs.append((str(k), float(v)))
                except (TypeError, ValueError):
                    continue
            pairs.sort(key=lambda x: x[1], reverse=True)
            if pairs:
                label = label or pairs[0][0]
                confidence = pairs[0][1]
                if len(pairs) > 1:
                    runner_label, runner_conf = pairs[1]
                break

    if confidence is None:
        for key in ("confidence", "probability", "answer_confidence"):
            try:
                if obj.get(key) is not None:
                    confidence = float(obj[key])
                    break
            except (TypeError, ValueError):
                pass

    if label not in QUESTION["criteria"]:
        label = "PARSE_REVIEW"
    return label, confidence, runner_label, runner_conf

def main():
    path = locate_dataset()
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    if "text" not in df.columns:
        raise KeyError(f"No 'text' column. Found: {list(df.columns)}")

    texts = df["text"].fillna("").astype(str).tolist()
    agent = thx01.load(MODEL_ID)  # Downloads model weights on first run.
    print(f"Loaded {MODEL_ID}; records={len(df)}; batch size={BATCH_SIZE}")

    # Preserve original row order, including empty records.
    result_rows = [None] * len(df)
    evaluated = [i for i, text in enumerate(texts) if text.strip()]
    started = time.time()

    for offset in range(0, len(evaluated), BATCH_SIZE):
        indices = evaluated[offset:offset + BATCH_SIZE]
        batch_texts = [texts[i] for i in indices]
        predictions = agent.predict_batch(
            batch_texts, {"domain": QUESTION}, batch_size=BATCH_SIZE
        )
        if isinstance(predictions, dict):
            predictions = predictions.get("results", predictions.get("outputs"))
        if predictions is None:
            raise RuntimeError("Unexpected predict_batch response. Inspect library output.")

        predictions = list(predictions)
        if len(predictions) != len(indices):
            raise RuntimeError(
                f"Output/input mismatch: {len(predictions)} outputs for {len(indices)} records."
            )

        for idx, text, raw in zip(indices, batch_texts, predictions):
            label, conf, runner, runner_conf = parse_choice(raw)
            result_rows[idx] = {
                "id": df.iloc[idx]["id"] if "id" in df.columns else idx,
                "text": text,
                "predicted_domain": label,
                "confidence": conf,
                "runner_up_domain": runner,
                "runner_up_confidence": runner_conf,
                "model": MODEL_ID,
                "raw_model_response": json.dumps(to_plain(raw), ensure_ascii=False),
            }

        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r for r in result_rows if r is not None]).to_csv(
            OUT, index=False, encoding="utf-8-sig"
        )
        print(f"Processed {min(offset + len(indices), len(evaluated))}/{len(evaluated)} non-empty records")

    for i, text in enumerate(texts):
        if result_rows[i] is None:
            result_rows[i] = {
                "id": df.iloc[i]["id"] if "id" in df.columns else i,
                "text": text,
                "predicted_domain": "NOT_EVALUATED",
                "confidence": None,
                "runner_up_domain": "",
                "runner_up_confidence": None,
                "model": MODEL_ID,
                "raw_model_response": "",
            }

    final = pd.DataFrame(result_rows)
    final.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("\nDomain counts:")
    print(final["predicted_domain"].value_counts(dropna=False).to_string())
    print(f"\nSaved: {OUT}")
    print(f"Elapsed: {(time.time() - started) / 60:.1f} minutes")
    print("PARSE_REVIEW rows mean the response parser needs inspection; do not score them.")

if __name__ == "__main__":
    main()
