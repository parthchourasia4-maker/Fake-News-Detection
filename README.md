# Signal/Check

Signal/Check is a Streamlit application that classifies **news headlines** as likely Real or Fake using learned textual patterns. It is a headline-pattern classifier, not a fact-checking system.

It does not determine objective truth, verify claims, understand article context, or replace professional fact-checking.

## Demo

Run the application locally:

```bash
streamlit run app.py
```

Live demo: **Not available yet**

## What It Includes

- Single-headline prediction with a decision-score-derived confidence signal.
- Model Performance page with comparison charts, class metrics, baseline comparison, confusion matrix, and cross-validation.
- Feature Importance page derived from the saved Linear SVC coefficients and TF-IDF vocabulary.
- Batch CSV prediction with validation, row-level error handling, and downloadable results.
- Project Information page covering the dataset, approach, limitations, and responsible use.

## Dataset

The project uses the FakeNewsNet dataset of labeled news headlines:

| Split or class | Samples |
| --- | ---: |
| Total | 23,196 |
| Real | 17,441 (75.19%) |
| Fake | 5,755 (24.81%) |
| Training split | 18,556 (80%) |
| Test split | 4,640 (20%) |

The class ratio is approximately 3.03:1 Real to Fake. The model uses title text only.

## NLP Pipeline

1. Lowercase the headline.
2. Remove URLs and non-letter characters.
3. Normalize whitespace.
4. Fit TF-IDF on training data only.
5. Train and evaluate Logistic Regression, Multinomial Naive Bayes, and Linear SVC.

### TF-IDF Configuration

- N-grams: unigrams and bigrams `(1, 2)`
- Maximum features: 15,000
- Minimum document frequency: 2
- Maximum document frequency: 95%
- English stop words

### Model Selection

Linear SVC was selected because it achieved the highest Fake-class F1 on 5-fold stratified cross-validation performed on the training data only. The held-out test set was not used for model selection.

## Final Evaluation

The final test evaluation is stored in `evaluation_results.json`.

| Metric | Linear SVC result |
| --- | ---: |
| Accuracy | 84.85% |
| Balanced accuracy | 77.87% |
| ROC-AUC | 0.8645 |
| Fake precision | 71.83% |
| Fake recall | 64.03% |
| Fake F1 | 67.71% |
| Real precision | 88.54% |
| Real recall | 91.72% |
| Real F1 | 90.10% |

### Confusion Matrix

Rows are actual labels and columns are predicted labels. Label order is Fake, Real.

```text
                 Predicted Fake    Predicted Real
Actual Fake              737               414
Actual Real              289             3,200
```

For the Fake class, 737 headlines were correctly identified and 414 Fake headlines were classified as Real. These are Fake-class false negatives, corresponding to a 36.0% miss rate among Fake test headlines.

### Majority Baseline

The majority baseline always predicts Real:

- Baseline accuracy: 75.19%
- Final model accuracy: 84.85%
- Improvement: +9.66 percentage points

### Cross-Validation

Five-fold `StratifiedKFold` cross-validation is performed on the training data only. TF-IDF is fitted inside each fold to prevent fold-level vocabulary or IDF leakage.

| Metric | Mean +/- standard deviation |
| --- | ---: |
| Accuracy | 83.51% +/- 0.63% |
| Macro F1 | 76.43% +/- 0.63% |
| Fake F1 | 63.52% +/- 0.81% |
| Balanced accuracy | 74.91% +/- 0.39% |

The standard deviation indicates variation across folds. The values above are formatted for display; full-precision values are retained in `evaluation_results.json`.

## Interpretability

The Feature Importance page displays the largest Linear SVC coefficients using the saved model and fitted vectorizer:

- Positive coefficients are associated with the Real class.
- Negative coefficients are associated with the Fake class.
- Larger absolute coefficients indicate stronger model-level influence.

These are statistical associations learned from this training dataset. They do not prove causality and do not prove that a word makes an article Fake or Real. The dataset contains strong celebrity and entertainment-related patterns, which may not generalize to other datasets or news domains.

## Responsible Use

Appropriate uses include research, education, analysis, portfolio demonstration, and assisting human reviewers.

Do not use this system for autonomous content removal, unsupervised fact-checking, regulatory decisions, mission-critical decisions, or any workflow without human oversight.

Reliable fact-checking requires multiple sources, contextual understanding, domain expertise, and human judgment.

## Installation and Local Use

Python 3.11 is specified in `runtime.txt`.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app loads `final_model.pkl`, `final_vectorizer.pkl`, and `evaluation_results.json` from the same directory as `app.py`.

To rerun the evaluation pipeline, place `FakeNewsNet.csv` beside `execute_pipeline.py`, or set a path explicitly:

```powershell
$env:FAKE_NEWS_DATA_PATH = "C:\path\to\FakeNewsNet.csv"
python execute_pipeline.py
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Include `app.py`, `requirements.txt`, `runtime.txt`, `evaluation_results.json`, `final_model.pkl`, and `final_vectorizer.pkl`.
3. In Streamlit Community Cloud, choose **Create app**.
4. Select the repository and branch, and set the main file to `app.py`.
5. Deploy and verify the five navigation pages.

The dataset is needed to rerun training, but not for normal inference from the saved artifacts. No live deployment URL is claimed here because one is not currently available.

## Project Structure

```text
fakenews/
|-- app.py
|-- execute_pipeline.py
|-- final_model.pkl
|-- final_vectorizer.pkl
|-- evaluation_results.json
|-- requirements.txt
|-- runtime.txt
|-- FakeNews_Production_Ready.ipynb
|-- FakeNewsNet.csv
`-- README.md
```

## License and Acknowledgment

This project is for educational and portfolio purposes. The dataset is FakeNewsNet. The modeling stack uses pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Altair, and Streamlit.
