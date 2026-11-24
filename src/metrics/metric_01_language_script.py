
"""
Metric 1: Language and Script Usage — V3

Purpose:
Evaluate language identification coverage, script-label consistency,
script representation compatibility, and Unicode character validity.

Important:
- Telugu words written in Latin script are classified as Telugu.
- English words are classified as English.
- Ambiguous and unknown words remain unclassified.
- Language distribution is calculated globally across all tokens.
- Script profile describes writing systems, not language mixing.
- Scores are heuristic estimates, not human-validated ground truth.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS AND SETTINGS
# ============================================================

DATASET_PATH = Path("data/raw/TelCo-QA-Dataset.xlsx")
OUTPUT_DIR = Path("outputs")

RECORDS_OUTPUT = OUTPUT_DIR / "metric_01_language_script_records.csv"
JSON_OUTPUT = OUTPUT_DIR / "metric_01_language_script.json"

WEIGHTS = {
    "language_identification_coverage": 0.30,
    "script_label_consistency": 0.30,
    "script_representation_compatibility": 0.20,
    "unicode_character_validity": 0.20,
}

HIGH_THRESHOLD = 80.0
MEDIUM_THRESHOLD = 60.0


# ============================================================
# LANGUAGE LEXICONS
# ============================================================

# Common Telugu words written in Latin script.
# Includes spelling and transliteration variations found in the dataset.

TELUGU_LATIN_WORDS = {
    # Common words and particles
    "ee", "idhi", "idi", "adi", "avi", "ivvi",
    "na", "naa", "ni", "nee", "naku", "naaku",
    "neeku", "manam", "mana", "meeru", "memu",
    "vallu", "vaallu", "nenu", "nuvvu",
    "ga", "lo", "ki", "ku", "tho", "to",
    "kosam", "gurinchi", "meeda", "paina",
    "valla", "valle", "nunchi", "tarvatha",
    "taruvatha", "mundhu", "lopala", "bayata",
    "ante", "anthe", "kuda", "kooda",
    "inka", "inkaa", "chala", "chaala",
    "konchem", "chaala", "pedda", "peddha",
    "manchi", "baaga", "bagaa", "sarigga",
    "nijanga", "simplega",

    # Common verbs and verb forms
    "chesaru", "chesaaru", "chesindi", "chesindhi",
    "chesthunnaru", "chestunnaru", "chestharu",
    "chestaru", "chestham", "chesthamani",
    "cheyyali", "cheyali", "cheyyalee",
    "cheyyadam", "cheyadam", "cheyyadaniki",
    "cheyadaniki", "cheppalante", "chepparu",
    "cheppadam", "cheppina", "cheppindi",
    "anipinchindhi", "anipinchindi",
    "anipistundi", "anipinchadam",
    "avuthundo", "avuthundi", "avuthunnayi",
    "avutundi", "avutunnayi", "avvali",
    "ayyindi", "ayyindhi", "ayindi", "ayindhi",
    "ayyaka", "ayyaka", "undhi", "undi",
    "unnayi", "unnaru", "ledu", "ledhu",
    "raaledu", "ivvali", "ivvandi",
    "nachindi", "nachindhi", "nachaledu",
    "dorakaledu", "jarigindi", "jarigindhi",
    "jaruguthundi", "jarugutundi",
    "vachindi", "vachindhi", "vastundi",
    "vastunnayi", "poyindi", "poyindhi",
    "chesukovali", "chesukovachu",
    "chudali", "chuddam", "chusthe",
    "chudandi", "maatladadu", "maatladaru",
    "nadusthondi", "nadustundi",
    "extend", "vadukoni", "vadhilaru",

    # Nouns, adjectives and expressions
    "rendu", "moodu", "okati", "oka",
    "mandi", "vishayam", "vishayalu",
    "ibbandi", "kastam", "kastanga",
    "baagundi", "bagundi", "baagundhi",
    "baagundhi", "manchidi", "ledu",
    "inka", "antha", "anthaa", "emi",
    "enti", "enduku", "ela", "eppudu",
    "ekkada", "ippudu", "appudu",
    "ninna", "repu", "ivala", "eeroju",
    "avunu", "kaadu", "kadu", "kakapothe",
    "aithe", "ayithe", "kabatti",
    "vachina", "vellina", "chesina",
    "cheyyalee", "avuthundo", "nunchi",
    "sambandhinchina", "sarigga", "nachindi",
    "vadhilaru", "vadukoni",

    # Additional common transliteration variants
    "inka", "inkem", "em", "emo",
    "manaki", "naaku", "naaku",
    "cheppalante", "nijamga", "nijanga",
    "peddha", "pedda", "konchem",
    "chesthunnaru", "chestunnaru",
}


# Common English words found in the dataset and general usage.

ENGLISH_WORDS = {
    # English words from the unknown-word inspection
    "opinion", "improve", "fans", "useful", "simple",
    "honestly", "important", "let's", "hope", "discuss",
    "final", "personally", "definitely", "announce",
    "overall", "release", "details", "official",
    "match", "update", "team", "movie", "schedule",
    "project", "website", "start", "clear", "publish",
    "date", "performance", "complete", "easy", "wait",
    "use", "department", "review", "exam", "version",
    "conference", "conduct", "registration", "plan",
    "issue", "response", "timetable", "list", "meeting",
    "share", "trailer", "placement", "ai", "features",
    "price", "scenes", "entertainment", "ticket",
    "next", "time", "better", "playing", "eleven",
    "changes", "product", "tomorrow", "afternoon",
    "playing", "project", "nuisance", "device",
    "availability", "new", "result", "marks", "portal",
    "check", "people", "common", "assembly", "session",
    "cinema", "audience", "experience", "positive",
    "phone", "explanation", "confusing", "last",
    "overs", "control", "miss", "costly", "api",
    "access", "account", "create", "privacy", "concern",
    "development", "approach", "photos", "online",
    "steps", "leaders", "form", "fill", "technology",
    "new", "changelog", "list", "captain", "post",
    "presentation", "local", "representatives",
    "results", "internship", "eligible", "branches",
    "department", "assignment", "submission",
    "friday", "notice", "library", "timings",
    "exams", "lab", "sessions", "explanation",
    "semester", "university", "upload", "kosam",
    "work", "emotional", "impact", "deserved",
    "congratulations", "kosam", "full", "technology",
    "new", "happy", "result", "held", "room",
    "development", "clear", "people", "positive",
    "important", "personally", "overall",

    # Common English function words
    "is", "he", "to", "am", "we", "or",
    "the", "a", "an", "and", "are", "was",
    "were", "be", "been", "being", "of",
    "in", "on", "at", "by", "for", "from",
    "with", "without", "as", "if", "but",
    "not", "no", "yes", "it", "its", "this",
    "that", "these", "those", "they", "them",
    "their", "there", "here", "what", "which",
    "who", "when", "where", "why", "how",
    "will", "would", "could", "should", "can",
    "may", "might", "must", "do", "does",
    "did", "done", "have", "has", "had",
    "i", "you", "your", "my", "me", "him",
    "her", "his", "our", "us", "all", "some",
    "any", "more", "most", "much", "many",
    "very", "just", "also", "only", "than",
    "then", "now", "so", "because", "while",
    "after", "before", "about", "up", "down",
    "out", "into", "over", "under", "again",
    "once", "here", "there", "each", "every",
    "both", "few", "other", "another", "same",
    "own", "such", "through", "during",
    "between", "against", "further",
}


# These are retained for future context-sensitive handling.
# The user's confirmed decisions classify the previously inspected
# short tokens as Telugu, so they are NOT treated as ambiguous here.

AMBIGUOUS_WORDS = set()


# ============================================================
# TOKENIZATION AND LANGUAGE IDENTIFICATION
# ============================================================

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)?|[\u0C00-\u0C7F]+"
)


def tokenize(text):
    """Extract Latin and Telugu-script word tokens."""
    if pd.isna(text):
        return []

    return TOKEN_PATTERN.findall(str(text))


def is_telugu_script(token):
    """Check whether a token contains Telugu Unicode characters."""
    return any("\u0C00" <= char <= "\u0C7F" for char in token)


def identify_token_language(token):
    """
    Identify a token as Telugu, English, Ambiguous, or Unknown.

    Telugu Unicode tokens are directly classified as Telugu.
    Latin tokens are checked against the lexicons.
    """
    normalized = token.lower().strip()

    if not normalized:
        return "Unknown"

    if is_telugu_script(normalized):
        return "Telugu"

    if normalized in AMBIGUOUS_WORDS:
        return "Ambiguous"

    if normalized in TELUGU_LATIN_WORDS:
        return "Telugu"

    if normalized in ENGLISH_WORDS:
        return "English"

    return "Unknown"


# ============================================================
# SCRIPT PROFILING
# ============================================================

def get_script_profile(text):
    """
    Determine the writing-system profile of a record.

    Romanized:
        Latin script only.

    Telugu script:
        Telugu Unicode script only.

    code-mixed:
        Both Latin and Telugu Unicode scripts.
    """
    if pd.isna(text):
        return "Not Evaluated"

    has_latin = False
    has_telugu = False

    for char in str(text):
        if "A" <= char <= "Z" or "a" <= char <= "z":
            has_latin = True

        elif "\u0C00" <= char <= "\u0C7F":
            has_telugu = True

    if has_latin and has_telugu:
        return "code-mixed"

    if has_telugu:
        return "Telugu script"

    if has_latin:
        return "Romanized"

    return "Not Evaluated"


def normalize_script_label(label):
    """Normalize existing dataset script labels."""
    if pd.isna(label):
        return "Not Evaluated"

    value = str(label).strip().lower()

    aliases = {
        "romanized": "Romanized",
        "romanised": "Romanized",
        "latin": "Romanized",
        "telugu": "Telugu script",
        "telugu script": "Telugu script",
        "code-mixed": "code-mixed",
        "codemixed": "code-mixed",
        "mixed": "code-mixed",
    }

    return aliases.get(value, "Not Evaluated")


# ============================================================
# CHARACTER VALIDITY AND SCRIPT COMPATIBILITY
# ============================================================

def calculate_unicode_validity(text):
    """
    Return the percentage of valid characters.

    ZWNJ and ZWJ are explicitly allowed.
    Normal whitespace, punctuation, symbols, digits,
    and Telugu combining marks are retained.
    """
    if pd.isna(text):
        return 0.0

    text = str(text)

    if not text:
        return 0.0

    valid = 0
    total = len(text)

    for char in text:
        category = unicodedata.category(char)

        if char in ("\u200c", "\u200d"):
            valid += 1

        elif category in ("Cc", "Cs", "Co", "Cn"):
            continue

        else:
            valid += 1

    return (valid / total) * 100.0


def calculate_script_compatibility(text):
    """
    Calculate the percentage of alphabetic characters
    that belong to Telugu or Latin scripts.
    """
    if pd.isna(text):
        return 0.0

    supported = 0
    alphabetic = 0

    for char in str(text):
        if char.isalpha():
            alphabetic += 1

            if (
                "A" <= char <= "Z"
                or "a" <= char <= "z"
                or "\u0C00" <= char <= "\u0C7F"
            ):
                supported += 1

    if alphabetic == 0:
        return 0.0

    return (supported / alphabetic) * 100.0


# ============================================================
# RECORD-LEVEL LANGUAGE IDENTIFICATION
# ============================================================

def calculate_record_language_metrics(text):
    tokens = tokenize(text)

    if not tokens:
        return {
            "telugu_token_percentage": 0.0,
            "english_token_percentage": 0.0,
            "ambiguous_token_percentage": 0.0,
            "unknown_token_percentage": 0.0,
            "language_identification_coverage": 0.0,
        }

    counts = Counter(
        identify_token_language(token)
        for token in tokens
    )

    total = len(tokens)

    telugu_pct = counts["Telugu"] / total * 100
    english_pct = counts["English"] / total * 100
    ambiguous_pct = counts["Ambiguous"] / total * 100
    unknown_pct = counts["Unknown"] / total * 100

    coverage = telugu_pct + english_pct

    return {
        "telugu_token_percentage": telugu_pct,
        "english_token_percentage": english_pct,
        "ambiguous_token_percentage": ambiguous_pct,
        "unknown_token_percentage": unknown_pct,
        "language_identification_coverage": coverage,
    }


# ============================================================
# QUALITY TIER
# ============================================================

def assign_quality_tier(score):
    if score >= HIGH_THRESHOLD:
        return "High"

    if score >= MEDIUM_THRESHOLD:
        return "Medium"

    return "Low"


# ============================================================
# MAIN METRIC EXECUTION
# ============================================================

def run_metric():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATASET_PATH)

    required_columns = {"text", "script"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    records = []

    # Global token counts: every token contributes equally.
    global_language_counts = Counter()

    # Global script-token counts.
    global_script_counts = Counter()

    # Record-level script profile counts.
    script_profile_counts = Counter()

    # Global component aggregation.
    total_records = len(df)

    for index, row in df.iterrows():
        text = row["text"]

        tokens = tokenize(text)

        language_counts = Counter(
            identify_token_language(token)
            for token in tokens
        )

        global_language_counts.update(language_counts)

        for token in tokens:
            if is_telugu_script(token):
                global_script_counts["Telugu script"] += 1
            else:
                global_script_counts["Latin"] += 1

        language_metrics = calculate_record_language_metrics(text)

        actual_profile = get_script_profile(text)
        declared_profile = normalize_script_label(row["script"])

        script_profile_counts[actual_profile] += 1

        if actual_profile == "Not Evaluated":
            label_consistency = 0.0
        else:
            label_consistency = (
                100.0
                if actual_profile == declared_profile
                else 0.0
            )

        script_compatibility = calculate_script_compatibility(text)
        unicode_validity = calculate_unicode_validity(text)

        record_score = (
            WEIGHTS["language_identification_coverage"]
            * language_metrics["language_identification_coverage"]
            +
            WEIGHTS["script_label_consistency"]
            * label_consistency
            +
            WEIGHTS["script_representation_compatibility"]
            * script_compatibility
            +
            WEIGHTS["unicode_character_validity"]
            * unicode_validity
        )

        record_tier = assign_quality_tier(record_score)

        records.append({
            "record_id": row.get("id", index + 1),
            "script_profile": actual_profile,
            "declared_script": declared_profile,
            **language_metrics,
            "script_label_consistency": label_consistency,
            "script_representation_compatibility": script_compatibility,
            "unicode_character_validity": unicode_validity,
            "overall_score": record_score,
            "quality_tier": record_tier,
        })

    results_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # GLOBAL LANGUAGE DISTRIBUTION
    # --------------------------------------------------------

    total_tokens = sum(global_language_counts.values())

    if total_tokens:
        language_distribution = {
            "telugu_token_percentage": (
                global_language_counts["Telugu"] / total_tokens * 100
            ),
            "english_token_percentage": (
                global_language_counts["English"] / total_tokens * 100
            ),
            "ambiguous_token_percentage": (
                global_language_counts["Ambiguous"] / total_tokens * 100
            ),
            "unknown_token_percentage": (
                global_language_counts["Unknown"] / total_tokens * 100
            ),
        }
    else:
        language_distribution = {
            "telugu_token_percentage": 0.0,
            "english_token_percentage": 0.0,
            "ambiguous_token_percentage": 0.0,
            "unknown_token_percentage": 0.0,
        }

    # --------------------------------------------------------
    # GLOBAL SCRIPT TOKEN DISTRIBUTION
    # --------------------------------------------------------

    total_script_tokens = sum(global_script_counts.values())

    if total_script_tokens:
        script_distribution = {
            "latin_token_percentage": (
                global_script_counts["Latin"]
                / total_script_tokens * 100
            ),
            "telugu_script_token_percentage": (
                global_script_counts["Telugu script"]
                / total_script_tokens * 100
            ),
            "mixed_script_token_percentage": 0.0,
        }
    else:
        script_distribution = {
            "latin_token_percentage": 0.0,
            "telugu_script_token_percentage": 0.0,
            "mixed_script_token_percentage": 0.0,
        }

    # --------------------------------------------------------
    # RECORD-LEVEL SCRIPT PROFILE DISTRIBUTION
    # --------------------------------------------------------

    if total_records:
        record_script_distribution = {
            profile: script_profile_counts[profile] / total_records * 100
            for profile in (
                "Romanized",
                "Telugu script",
                "code-mixed",
                "Not Evaluated",
            )
        }
    else:
        record_script_distribution = {}

    # --------------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------------

    component_columns = [
        "language_identification_coverage",
        "script_label_consistency",
        "script_representation_compatibility",
        "unicode_character_validity",
    ]

    component_scores = {
        column: float(results_df[column].mean())
        if not results_df.empty else 0.0
        for column in component_columns
    }

    overall_score = sum(
        component_scores[name] * weight
        for name, weight in WEIGHTS.items()
    )

    overall_tier = assign_quality_tier(overall_score)

    # --------------------------------------------------------
    # QUALITY TIER DISTRIBUTION
    # --------------------------------------------------------

    if total_records:
        tier_distribution = {
            tier: float(
                (results_df["quality_tier"] == tier).mean() * 100
            )
            for tier in ("High", "Medium", "Low")
        }
    else:
        tier_distribution = {
            "High": 0.0,
            "Medium": 0.0,
            "Low": 0.0,
        }

    # --------------------------------------------------------
    # SCRIPT-LABEL AGREEMENT
    # --------------------------------------------------------

    evaluated_records = results_df[
        results_df["script_profile"] != "Not Evaluated"
    ]

    if len(evaluated_records):
        script_label_agreement = float(
            evaluated_records["script_label_consistency"].mean()
        )
    else:
        script_label_agreement = 0.0

    # --------------------------------------------------------
    # SAVE RECORD-LEVEL OUTPUT
    # --------------------------------------------------------

    percentage_columns = [
        column for column in results_df.columns
        if column.endswith("_percentage")
        or column.endswith("_consistency")
        or column.endswith("_compatibility")
        or column.endswith("_validity")
        or column == "overall_score"
    ]

    results_df[percentage_columns] = (
        results_df[percentage_columns].round(2)
    )

    results_df.to_csv(RECORDS_OUTPUT, index=False)

    # --------------------------------------------------------
    # SAVE JSON SUMMARY
    # --------------------------------------------------------

    summary = {
        "metric": "Metric 1: Language and Script Usage",
        "version": "V3",
        "dataset_records": total_records,
        "methodology": {
            "language_distribution": (
                "Global token-level distribution. "
                "Each token contributes equally."
            ),
            "script_distribution": (
                "Token-level writing-system distribution."
            ),
            "record_script_profile": (
                "Record-level Unicode writing-system profile. "
                "This does not identify bilingual language mixing "
                "when both languages use Latin script."
            ),
            "scoring": (
                "Heuristic weighted composite; not human-validated."
            ),
            "weights": WEIGHTS,
        },
        "language_distribution": language_distribution,
        "script_distribution": script_distribution,
        "record_script_profile_distribution": (
            record_script_distribution
        ),
        "component_scores": component_scores,
        "script_label_agreement": script_label_agreement,
        "overall_score": overall_score,
        "overall_quality_tier": overall_tier,
        "quality_tier_distribution": tier_distribution,
        "interpretation_notes": [
            "Unknown tokens are not automatically assigned a language.",
            "Language identification coverage depends on lexicon coverage.",
            "Script-label agreement checks writing-system labels, "
            "not linguistic correctness.",
            "Script representation compatibility measures support "
            "for Telugu and Latin alphabetic characters.",
            "Unicode validity is a technical character-validity check, "
            "not a measure of linguistic quality.",
            "The quality score and thresholds are heuristic."
        ],
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # CONSOLE REPORT
    # --------------------------------------------------------

    def print_section(title):
        print("\n" + title)
        print("-" * 55)

    def print_percentages(data):
        for name, value in data.items():
            print(f"{name:45s}: {value:.2f}%")

    print(f"\nDataset records: {total_records}")

    print_section("LANGUAGE DISTRIBUTION")
    print_percentages(language_distribution)

    print_section("SCRIPT DISTRIBUTION")
    print_percentages(script_distribution)

    print_section("RECORD SCRIPT PROFILE DISTRIBUTION")
    print_percentages(record_script_distribution)

    print_section("COMPONENT SCORES")
    print_percentages(component_scores)

    print(f"\nScript-label agreement: {script_label_agreement:.2f}%")

    print_section("QUALITY TIER DISTRIBUTION")
    print_percentages(tier_distribution)

    print(f"\nOVERALL SCORE: {overall_score:.2f}%")
    print(f"QUALITY TIER: {overall_tier}")

    print("\nOUTPUT FILES")
    print(f"Records: {RECORDS_OUTPUT}")
    print(f"Summary: {JSON_OUTPUT}")


if __name__ == "__main__":
    run_metric()
