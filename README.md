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
## Metric 4 � Annotation Reliability (In Progress) - 24-11-2026

- **Objective:** Verify the availability and reliability of existing dataset annotations using Yama.
- **Annotation availability:** The dataset contains sentiment and topic labels.
- **Verification approach:** Yama independently predicts labels from the text, and its predictions are compared against the existing annotations.
- **Sample evaluation:** Tested 25 records for sentiment and 25 records for topic.
- **Yama prediction coverage:**
  - Sentiment: 15/25 predictions (60%); 10 blank predictions.
  - Topic: 17/25 predictions (68%); 8 blank predictions.
  - Combined: 32/50 predictions (64%); 18 blank predictions.
- **Observation:** Yama's non-blank predictions appeared reasonably accurate during the limited sample inspection. However, several predictions were returned blank, requiring further investigation.
- **Current status:** Sample testing completed. Prediction extraction and blank-output handling need to be investigated before full-dataset evaluation.
- **Data integrity:** Existing annotations are preserved. Disagreements are treated as review candidates and are not automatically corrected.
- **Scoring approach:** Annotation availability and Yama prediction coverage are reported separately. Agreement is calculated among non-blank predictions. Datasets without applicable annotations receive a score of 0 for this metric and are excluded from the overall quality score aggregation.


## Progress Update - 25 November 2026

### Metric 4 - Annotation Reliability
- Implemented Yama-based annotation verification using the LatentAILabs/yama-indic model.
- Detected annotation availability across the sentiment and topic columns.
- Updated the evaluation to calculate reliability using only non-abstained Yama predictions.
- Abstentions are excluded from the reliability score denominator.
- Tested the metric on 50 records per annotation column (100 annotation cells).
- Sample results: 88.0% prediction coverage, 80.68% agreement among predictions, 12 abstentions, and 0 unparsed responses.
- Preserved the original dataset annotations; mismatches remain candidates for manual review.
- Sample runs are excluded from the overall dataset quality score.

