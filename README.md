Metric 05 — Target-Domain Representativeness
Purpose
Metric 05 estimates the domain composition of the Telugu-English code-mixed dataset. It classifies each non-empty sentence into one of seven topic categories and reports the percentage of records assigned to a selected target domain, along with the full predicted domain distribution.

The seven categories are:
- General
- Cricket & Sports
- Education & Students
- Technology & AI
- Politics & Current Affairs
- Movies & Entertainment
- Multiple topics
How the approach evolved
1. Multilingual embedding similarity (initial prototype)
The first prototype used intfloat/multilingual-e5-base to embed each sentence and a user-provided target-domain description. Cosine similarity and a configurable threshold determined whether a record was relevant.

This approach was investigated using Sports as the target domain. The result was only 3 relevant records out of 7,223 at the selected threshold. Reviewing the saved record-level results showed that clearly sports-related sentences often had similarity margins just below the cutoff. Lowering the cutoff increased the number of selected records, but also admitted general or template-like sentences.

Decision: Do not use a single embedding-similarity threshold as the final domain classifier. The threshold was too sensitive and did not reliably distinguish the target domain from general conversational content.
2. Supervised TF-IDF + Logistic Regression experiment
The dataset already contained topic labels, so a supervised baseline was trained using:
- Input: text
- Target: topic
- Features: word TF-IDF and character TF-IDF n-grams
- Classifier: Logistic Regression
- Split: group-aware train/test split using normalized text to reduce duplicate leakage
The experiment used 5,795 training records and 1,428 test records. It achieved 98.88% test accuracy against the existing topic labels.

However, an interactive test exposed a limitation: a clearly movie-related opinion written in informal romanized Telugu was predicted as General. A sports sentence with explicit vocabulary was classified correctly.

Interpretation: The high test accuracy measured agreement with the dataset's existing labels. It did not independently establish that the model understood every sentence's domain correctly. The model could reproduce existing label patterns and mistakes.

Decision: Keep this classifier as an experiment/baseline, but do not treat its test accuracy as proof of domain correctness.
3. Pretrained multilingual zero-shot classification (current prototype)
The current Metric 05 prototype uses a pretrained multilingual zero-shot sequence-classification model:

Horizon-Labs/multilingualzeroshot-small

The model compares each sentence with descriptive candidate labels for the seven domains. It assigns the highest-scoring category without requiring additional labelled training examples.

For the selected target domain, the metric counts records predicted into that category and calculates:

Target-domain share (%) = records predicted in target domain / non-empty evaluated records × 100

The record-level output includes the predicted domain, prediction score, and scores for each candidate category. The report also includes the predicted distribution across all seven categories and, when available, a diagnostic comparison with the existing topic labels.
How to run
From the repository root, install the required packages in the project's virtual environment:
pip install transformers torch
Run Metric 05 for Sports:
python src/metrics/metric_05_domain_representativeness.py --domain "Cricket & Sports"
To evaluate another supported category, replace the domain value with one of the seven category names.

The pretrained model is downloaded and cached on its first run. Subsequent runs can use the local cache.
Outputs
The script writes domain-specific outputs under outputs/:
- metric_05_zero_shot_<domain>_records.csv — record-level predicted categories and scores
- metric_05_zero_shot_<domain>_report.json — summary, predicted domain distribution, and limitations
Interpretation and limitations
This is a model-estimated domain composition. It is not, by itself, proof of statistical representativeness of a real-world population. A representativeness claim requires a defined reference population or expected domain distribution.

Zero-shot model scores are not calibrated probabilities of correctness. Performance may be affected by informal romanized Telugu, spelling variation, slang, short sentences, candidate-label wording, and sentences that cover multiple domains. The current prototype uses single-label classification, so each record receives one main category.

Predictions should be inspected on representative examples before the metric is treated as validated. The existing topic labels are useful for comparison but are not independent ground truth.
Current status
Metric 05 is a zero-shot prototype. Its outputs are suitable for exploratory domain-distribution analysis and later integration testing. Validation and final aggregation decisions remain separate from the classifier's raw predictions.