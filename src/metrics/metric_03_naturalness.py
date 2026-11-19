
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# =========================================================
# PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data" / "raw" / "TelCo-QA-Dataset.xlsx"

RECORD_REPORT_PATH = (
    ROOT / "outputs" / "metric_03_naturalness_records.csv"
)

SUMMARY_PATH = ROOT / "outputs" / "metric_03_naturalness.json"


WEIGHTS = {
    "language_identification_coverage": 0.20,
    "switching_stability": 0.20,
    "script_compatibility": 0.20,
    "repetition": 0.20,
    "text_structure": 0.20
}


# =========================================================
# LANGUAGE VOCABULARY
# =========================================================

ENGLISH_WORDS = set("""
a an the and or but if then because so as at by for from
in into of on onto to with without about after before between
through during over under again further once here there when
where why how all any both each few more most other some such
no nor not only own same than too very can could should would
will just also still already really quite rather much many
is am are was were be been being do does did have has had
having i me my mine we us our ours you your yours he him his
she her hers it its they them their theirs this that these those
what which who whom whose

good bad better best worse worst great nice useful important
simple clear difficult easy different new old first last next
time day week month year people person friend family
think know want need make made get got go went come came
see saw say said tell told use used using work worked working
like liked love loved hate hated feel felt seem seems
app phone mobile computer system software update
problem issue error bug feature performance quality
movie film actor actress director story scene music song
game match team player captain cricket sports
food taste restaurant hotel place city road
student teacher school college university exam class
book notes subject topic concept explain explanation
government party election leader public policy
technology internet online website data network
excellent amazing awesome fantastic
today tomorrow yesterday now later soon
yes okay ok please thanks thank

improve fans honestly personally let's hope discuss final
announce overall release details official schedule project
start publish date complete wait department review version
conference conduct registration plan response timetable
list meeting share trailer placement ai features price
scenes session decision update stable bugs
account access create target approach press committee
report submit deadline fix director effort background
presentation points innings pressure uncertainty
implementation results positive decisions changes
interest maintain maintaining experience
performance smooth release official announcements
network storage application interface beginners
problems crashing frustrating crashing
""".split())


ROMAN_TELUGU_WORDS = set("""
nenu naaku naa na
nuvvu nee neeku
meeru meeku mee mana manam
vaadu vaallu vallu aame ame
idhi idi adhi adi ee aa
enti em ela enduku eppudu ekkada
untundi untundhi unnanu unnadu undi undhi
unnayi unnaru ledu kaadu kadu avunu
chesanu chesadu chesaru chesindi chesindhi
chesaaru chesthamani chesthunnanu
chesthunnaru chesthundi chesthunnadi
chesina cheyyali cheyali cheyyachu
cheyyadam cheyyadaniki cheyyalee
cheppanu cheppadu chepparu cheppali
cheppalante annaru antaru anipisthundi
anipinchindi anipinchindhi
vastundi vasthundi vachindi vachindhi
vellali vellanu velladu velthunnadu
avutundi avuthundi ayindi ayindhi
ayyindi ayyindhi ayyaka
kavali kaavali
chala chaala baaga baagaa manchi manchidi
bagundi baagundi baagundhi
konchem koncham inka inkaa inkem
kuda kooda matrame maatrame
valla valle kosam gurinchi
meeda mida nundi nunchi
tho thone lo ki ku ni
ante kani kabatti
aithe ayithe
tarvatha taruvatha mundu
ippudu appudu
ikkada akkada ikkadiki akkadiki
ani gaa ga
rendu moodu nalugu
oka okati
naku naaku
nachindi nachindhi
ardham artham
kastam kashtam
sare avasaram
nijanga nijam
simple ga
opinion lo
cheppalante
chuddam chooddam chusthe
kanipisthundi kanipinchindi
jarigindi jarigindhi
poyindi
avvali anthe
mandi
ayyindi ayyindhi
cheyyadaniki
chesthamani
chesthadu chesthundi
chestharu chestaru
avuthundo avuthundi
avuthunnayi avuthunnadu
avuthunnaru
avuthundhi
avvaledu
avvakapothe
avvachu
chesukovali
chesukunnaru
chesukunnadu
chesukundi
chesukunnaru
undachu undochu
ledu ledu
emaina
endhuku enduku
elaanti
elanti
vishayam
vishayalu
andaru
anni
antha
anthe
inka
inka
manchi
baaga
chaala
konchem
kontha
maatladadu
maatladaru
chepparu
cheppadu
cheppindi
cheppindhi
cheppina
cheppina
avuthundi
avuthundhi
chesaru
chesaaru
chesthunnaru
chesthunnadu
avuthundi
ayindi
ayyindi
ayindhi
chala
chaala
tharvatha
tarvatha
nunchi
nundi
meeda
mida
valla
valle
kosam
gurthundi
gurthu
naku
naaku
mee
mana
manchi
kastam
kashtam
sensible
rendu
moodu
okati
""".split())


# =========================================================
# TOKENIZATION
# =========================================================

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u0C00-\u0C7F]+|\d+|[^\w\s]",
    re.UNICODE
)

TELUGU_PATTERN = re.compile(r"[\u0C00-\u0C7F]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


def tokenize(text):
    """Extract words, numbers and punctuation."""

    return TOKEN_PATTERN.findall(str(text))


# =========================================================
# LANGUAGE IDENTIFICATION
# =========================================================

def identify_token(token):
    """
    Identify a token as Telugu, English, mixed,
    ambiguous, number, punctuation or unknown.
    """

    has_telugu = bool(TELUGU_PATTERN.search(token))
    has_latin = bool(LATIN_PATTERN.search(token))

    if has_telugu and has_latin:
        return "mixed"

    if has_telugu:
        return "telugu"

    if token.isdigit():
        return "number"

    if not has_latin:
        return "punctuation"

    normalized = token.lower()

    is_english = normalized in ENGLISH_WORDS
    is_roman_telugu = normalized in ROMAN_TELUGU_WORDS

    if is_english and is_roman_telugu:
        return "ambiguous"

    if is_english:
        return "english"

    if is_roman_telugu:
        return "telugu"

    return "unknown"


# =========================================================
# COMPONENT 1: LANGUAGE IDENTIFICATION COVERAGE
# =========================================================

def calculate_language_coverage(classifications):
    """
    Measures the percentage of alphabetic tokens
    that can be identified confidently.

    Telugu script, Romanized Telugu and English
    are recognized independently.

    Mixed-script tokens are also identified.
    """

    language_labels = {
        "telugu",
        "english",
        "mixed",
        "ambiguous",
        "unknown"
    }

    language_tokens = [
        label for label in classifications
        if label in language_labels
    ]

    if not language_tokens:
        return 0.0

    identified = sum(
        label in {"telugu", "english", "mixed"}
        for label in language_tokens
    )

    return round(
        identified / len(language_tokens) * 100,
        2
    )


# =========================================================
# COMPONENT 2: SWITCHING STABILITY
# =========================================================

def calculate_switching_stability(classifications):
    """
    Measures excessive back-and-forth language switching.

    Unknown and ambiguous words break a sequence rather
    than being silently ignored.

    A normal single Telugu-English transition is allowed.
    """

    segments = []
    current_segment = []

    for label in classifications:

        if label in {"telugu", "english"}:
            current_segment.append(label)

        elif label in {
            "unknown",
            "ambiguous",
            "mixed"
        }:
            if current_segment:
                segments.append(current_segment)
                current_segment = []

    if current_segment:
        segments.append(current_segment)

    rapid_switches = 0
    comparable_positions = 0

    for segment in segments:

        if len(segment) < 3:
            continue

        for index in range(len(segment) - 2):

            a = segment[index]
            b = segment[index + 1]
            c = segment[index + 2]

            comparable_positions += 1

            if a == c and a != b:
                rapid_switches += 1

    if comparable_positions == 0:
        return 100.0

    score = (
        1 - rapid_switches / comparable_positions
    ) * 100

    return round(max(0.0, score), 2)


# =========================================================
# COMPONENT 3: SCRIPT COMPATIBILITY
# =========================================================

def calculate_script_compatibility(text):
    """
    Evaluates whether individual whitespace-delimited
    word units contain an unexpected mixture of Telugu
    and Latin characters.

    Switching scripts between separate words is allowed.
    """

    word_units = str(text).split()

    alphabetic_units = [
        unit for unit in word_units
        if LATIN_PATTERN.search(unit)
        or TELUGU_PATTERN.search(unit)
    ]

    if not alphabetic_units:
        return 100.0

    incompatible = 0

    for unit in alphabetic_units:

        has_telugu = bool(TELUGU_PATTERN.search(unit))
        has_latin = bool(LATIN_PATTERN.search(unit))

        if has_telugu and has_latin:
            incompatible += 1

    score = (
        1 - incompatible / len(alphabetic_units)
    ) * 100

    return round(max(0.0, score), 2)


# =========================================================
# COMPONENT 4: REPETITION
# =========================================================

def calculate_repetition(tokens):
    """
    Penalizes consecutive repeated words.

    Ordinary repetition in different parts of a sentence
    is not automatically considered an error.
    """

    words = [
        token.lower()
        for token in tokens
        if LATIN_PATTERN.search(token)
        or TELUGU_PATTERN.search(token)
    ]

    if len(words) < 2:
        return 100.0

    repeated_pairs = sum(
        words[index] == words[index + 1]
        for index in range(len(words) - 1)
    )

    repetition_ratio = (
        repeated_pairs / (len(words) - 1)
    )

    score = 100 * (1 - repetition_ratio)

    return round(max(0.0, score), 2)


# =========================================================
# COMPONENT 5: TEXT STRUCTURE
# =========================================================

def calculate_text_structure(text):
    """
    Checks basic formatting problems:
    - Repeated whitespace
    - Spaces before punctuation
    - Repeated punctuation
    - Missing separation after punctuation
    """

    text = str(text)

    if not text.strip():
        return 0.0

    penalties = 0

    if re.search(r"\s{2,}", text):
        penalties += 1

    if re.search(r"\s+[,.!?;:]", text):
        penalties += 1

    if re.search(r"[,.!?;:]{2,}", text):
        penalties += 1

    if re.search(r"[,.!?;:][A-Za-z0-9]", text):
        penalties += 1

    score = 100 - penalties * 25

    return round(max(0.0, score), 2)


# =========================================================
# SENTENCE EVALUATION
# =========================================================

def evaluate_sentence(text):
    """Calculate all five components for one record."""

    if pd.isna(text) or not str(text).strip():

        return {
            "language_identification_coverage": None,
            "switching_stability": None,
            "script_compatibility": None,
            "repetition": None,
            "text_structure": None,
            "naturalness_score": None,
            "identified_telugu_tokens": 0,
            "identified_english_tokens": 0,
            "unknown_tokens": 0,
            "ambiguous_tokens": 0,
            "language_switches": 0,
            "unknown_words": "",
            "evaluation_status": "Empty text"
        }

    text = str(text)

    tokens = tokenize(text)

    classifications = [
        identify_token(token)
        for token in tokens
    ]

    telugu_count = classifications.count("telugu")
    english_count = classifications.count("english")
    unknown_count = classifications.count("unknown")
    ambiguous_count = classifications.count("ambiguous")

    language_sequence = [
        label
        for label in classifications
        if label in {"telugu", "english"}
    ]

    language_switches = sum(
        language_sequence[index]
        != language_sequence[index + 1]
        for index in range(len(language_sequence) - 1)
    )

    unknown_words = [
        token.lower()
        for token, label in zip(tokens, classifications)
        if label == "unknown"
    ]

    scores = {
        "language_identification_coverage":
            calculate_language_coverage(classifications),

        "switching_stability":
            calculate_switching_stability(classifications),

        "script_compatibility":
            calculate_script_compatibility(text),

        "repetition":
            calculate_repetition(tokens),

        "text_structure":
            calculate_text_structure(text)
    }

    overall_score = sum(
        scores[name] * WEIGHTS[name]
        for name in WEIGHTS
    )

    return {
        **scores,

        "naturalness_score": round(overall_score, 2),

        "identified_telugu_tokens": telugu_count,
        "identified_english_tokens": english_count,
        "unknown_tokens": unknown_count,
        "ambiguous_tokens": ambiguous_count,
        "language_switches": language_switches,

        "unknown_words": ", ".join(unknown_words),

        "evaluation_status": "Evaluated"
    }


# =========================================================
# QUALITY TIERS
# =========================================================

def assign_quality_tier(score):

    if pd.isna(score):
        return "Not Evaluated"

    if score >= 80:
        return "High"

    if score >= 60:
        return "Medium"

    return "Low"


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 65)
    print("METRIC 03 - CODE-MIX NATURALNESS")
    print("=" * 65)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_excel(DATASET_PATH)

    if "text" not in dataset.columns:
        raise ValueError(
            "Dataset must contain a 'text' column."
        )

    print(f"\nDataset records: {len(dataset)}")

    # Evaluate all records.
    evaluations = [
        evaluate_sentence(text)
        for text in dataset["text"]
    ]

    evaluation_df = pd.DataFrame(evaluations)

    results = pd.concat(
        [
            dataset.reset_index(drop=True),
            evaluation_df
        ],
        axis=1
    )

    results["naturalness_tier"] = results[
        "naturalness_score"
    ].apply(assign_quality_tier)

    valid_scores = results[
        "naturalness_score"
    ].dropna()

    if valid_scores.empty:
        raise ValueError(
            "No non-empty records available for evaluation."
        )

    overall_score = round(valid_scores.mean(), 2)

    component_summary = {
        component: round(
            results[component].mean(),
            2
        )
        for component in WEIGHTS
    }

    tier_distribution = {
        tier: int(
            (results["naturalness_tier"] == tier).sum()
        )
        for tier in [
            "High",
            "Medium",
            "Low",
            "Not Evaluated"
        ]
    }

    # Collect dataset-level language statistics.
    total_telugu = int(
        results["identified_telugu_tokens"].sum()
    )

    total_english = int(
        results["identified_english_tokens"].sum()
    )

    total_unknown = int(
        results["unknown_tokens"].sum()
    )

    total_ambiguous = int(
        results["ambiguous_tokens"].sum()
    )

    unknown_word_frequency = Counter()

    for value in results["unknown_words"].dropna():

        for word in str(value).split(", "):
            if word:
                unknown_word_frequency[word] += 1

    top_unknown_words = [
        {
            "word": word,
            "frequency": count
        }
        for word, count in unknown_word_frequency.most_common(30)
    ]

    # Create output directories.
    RECORD_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save record-level report.
    results.to_csv(
        RECORD_REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Build JSON summary.
    summary = {
        "metric": "Code-Mix Naturalness",
        "version": "V1",

        "overall_score": overall_score,

        "quality_tier": assign_quality_tier(
            overall_score
        ),

        "total_records": len(dataset),
        "evaluated_records": int(len(valid_scores)),

        "empty_records": int(
            results["naturalness_score"].isna().sum()
        ),

        "component_scores": component_summary,
        "component_weights": WEIGHTS,

        "quality_tier_distribution": tier_distribution,

        "language_statistics": {
            "identified_telugu_tokens": total_telugu,
            "identified_english_tokens": total_english,
            "unknown_tokens": total_unknown,
            "ambiguous_tokens": total_ambiguous
        },

        "top_unrecognized_words": top_unknown_words,

        "method": (
            "Equal-weight heuristic evaluation using "
            "language identification coverage, switching "
            "stability, script compatibility, repetition "
            "and text structure."
        ),

        "monolingual_policy": (
            "Monolingual Telugu and English sentences are "
            "eligible for high scores. Bilingual composition "
            "is not mandatory."
        ),

        "language_identification_note": (
            "Telugu script is identified using Unicode. "
            "Romanized Telugu and English are identified "
            "using explicit word lexicons. Unknown and "
            "ambiguous words are reported separately."
        ),

        "limitations": [
            "This is a heuristic evaluation, not a "
            "human-validated measure of linguistic naturalness.",

            "Romanized Telugu and English vocabulary "
            "coverage is incomplete.",

            "Semantic coherence and grammatical correctness "
            "are not directly evaluated.",

            "Switching stability is a simplified indicator "
            "and does not establish whether a language switch "
            "is linguistically natural.",

            "Script compatibility evaluates mixed-script "
            "characters within individual whitespace-delimited "
            "tokens, not language switching between words.",

            "Component weights and quality-tier thresholds "
            "are provisional design choices."
        ]
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

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n" + "-" * 65)
    print("COMPONENT SCORES")
    print("-" * 65)

    for component, score in component_summary.items():

        print(
            f"{component.replace('_', ' ').title():38s}: "
            f"{score:.2f}%"
        )

    print("\n" + "-" * 65)
    print("QUALITY TIER DISTRIBUTION")
    print("-" * 65)

    for tier, count in tier_distribution.items():

        print(f"{tier:20s}: {count}")

    print("\n" + "-" * 65)
    print("LANGUAGE IDENTIFICATION STATISTICS")
    print("-" * 65)

    print(f"Telugu tokens     : {total_telugu}")
    print(f"English tokens    : {total_english}")
    print(f"Unknown tokens    : {total_unknown}")
    print(f"Ambiguous tokens  : {total_ambiguous}")

    print("\n" + "=" * 65)

    print(
        f"OVERALL NATURALNESS SCORE: "
        f"{overall_score:.2f}%"
    )

    print(
        f"OVERALL QUALITY TIER: "
        f"{assign_quality_tier(overall_score)}"
    )

    print("=" * 65)

    print(f"\nRecord report saved: {RECORD_REPORT_PATH}")
    print(f"Summary saved: {SUMMARY_PATH}")

    print("\nNOTE:")
    print(
        "This is a heuristic evaluation, not a "
        "human-validated naturalness measurement."
    )


if __name__ == "__main__":
    main()
