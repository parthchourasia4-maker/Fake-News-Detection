# Fake News Detector

A machine learning and NLP application that classifies news headlines as
likely **Real** or **Fake** based on learned textual patterns.

> **Important:** This is a text-pattern classification system, not a fact-checking
> engine. It does not verify claims, determine objective truth, understand full
> article context, or replace professional fact-checking.

## Live Demo

**Try the deployed application:**

https://fake-news-detection-vm6opb3atdc8itn4qzbnli.streamlit.app/

## Project Highlights

- End-to-end NLP classification pipeline
- TF-IDF feature extraction using unigrams and bigrams
- Comparison of Logistic Regression, Multinomial Naive Bayes, and Linear SVC
- Stratified cross-validation performed without test-set leakage
- Class-imbalance-aware evaluation
- Confusion matrix, ROC-AUC, balanced accuracy, and per-class metrics
- Interactive Streamlit dashboard
- Single-headline prediction
- Batch CSV prediction with downloadable results
- Model coefficient-based feature interpretation

## Why This Project?

Fake-news classification is a useful example of a real-world NLP problem
where accuracy alone can be misleading.

The dataset is imbalanced toward Real headlines, so this project evaluates
the model using multiple metrics instead of relying only on accuracy.

The workflow emphasizes:
- Correct train/test separation
- Prevention of feature-extraction leakage
- Stratified validation
- Fake-class performance
- Baseline comparison
- Error interpretation
- Responsible use of predictions

## Dataset

The project uses labeled news headlines from the **FakeNewsNet** dataset.

| Split / Class | Samples |
|---|---:|
| Total | 23,196 |
| Real | 17,441 (75.19%) |
| Fake | 5,755 (24.81%) |
| Training | 18,556 (80%) |
| Test | 4,640 (20%) |

The dataset contains approximately **3.03 times more Real headlines than
Fake headlines**.

The model uses **headline/title text only**.

## NLP Pipeline

The text-processing workflow is:

1. Convert headlines to lowercase.
2. Remove URLs.
3. Remove non-letter characters.
4. Normalize whitespace.
5. Split the data into training and test sets.
6. Fit TF-IDF using training data only.
7. Train multiple classification algorithms.
8. Compare models using validation metrics.
9. Select the final model.
10. Evaluate once on the held-out test set.

This separation prevents information from the test set from influencing
feature extraction or model selection.

## TF-IDF Configuration

| Parameter | Value |
|---|---|
| N-gram range | (1, 2) |
| Maximum features | 15,000 |
| Minimum document frequency | 2 |
| Maximum document frequency | 95% |
| Stop words | English |
## Model Selection

Three supervised learning approaches were evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC

**Linear SVC** was selected as the final model based on the Fake-class
performance during stratified cross-validation on the training data.

The held-out test set was not used for model selection.

## Final Test Performance

The final evaluation is stored in `evaluation_results.json`.

| Metric | Linear SVC |
|---|---:|
| Accuracy | **84.85%** |
| Balanced Accuracy | **77.87%** |
| ROC-AUC | **0.8645** |
| Fake Precision | **71.83%** |
| Fake Recall | **64.03%** |
| Fake F1 | **67.71%** |
| Real Precision | **88.54%** |
| Real Recall | **91.72%** |
| Real F1 | **90.10%** |

### Baseline Comparison

A majority-class classifier that always predicts Real achieves **75.19%**
accuracy.

The final Linear SVC reaches **84.85%**, an improvement of **9.66 percentage
points** over that baseline.

This comparison is important because accuracy alone could otherwise make an
imbalanced classifier appear stronger than it really is.

## Confusion Matrix

Rows represent actual labels and columns represent predictions.

| | Predicted Fake | Predicted Real |
|---|---:|---:|
| Actual Fake | 737 | 414 |
| Actual Real | 289 | 3,200 |

The Fake-class recall of 64.03% means the model misses a meaningful portion
of Fake headlines. This limitation should be considered before using the
system in any real-world workflow.
## Cross-Validation

Five-fold `StratifiedKFold` cross-validation is performed using the training
data only. TF-IDF is fitted independently inside each fold to prevent
fold-level vocabulary and IDF leakage.

| Metric | Mean +/- Standard Deviation |
|---|---:|
| Accuracy | 83.51% +/- 0.63% |
| Macro F1 | 76.43% +/- 0.63% |
| Fake F1 | 63.52% +/- 0.81% |
| Balanced Accuracy | 74.91% +/- 0.39% |

## Streamlit Application

The deployed application provides:

### Single Prediction
Enter a news headline and receive a Real/Fake classification with a
decision-score-derived confidence signal.

### Model Performance
Explore accuracy, class-level metrics, baseline comparison, confusion matrix,
ROC-AUC, and cross-validation results through interactive visualizations.

### Feature Importance
Inspect influential TF-IDF terms using the learned Linear SVC coefficients.

### Batch Prediction
Upload a CSV containing headlines, validate the input, generate predictions,
and download the results.

### Project Information
Review the dataset, methodology, limitations, and responsible-use guidance.

## Interpretability

The feature-importance view shows terms with strong Linear SVC coefficients.

These coefficients represent statistical associations learned from the
training data. They do not establish causation and do not prove that a
particular word makes a headline Fake or Real.

Dataset-specific patterns may also fail to generalize to other news domains.

## Responsible Use

This project is intended for education, research, experimentation, and
portfolio demonstration.

It should **not** be used for autonomous content removal, definitive
fact-checking, regulatory decisions, or other high-impact decisions without
human review.

Reliable fact-checking requires contextual understanding, multiple sources,
domain expertise, and human judgment.

## Tech Stack

**Python | pandas | NumPy | scikit-learn | Matplotlib | Seaborn | Altair |
Streamlit | Jupyter Notebook | Git/GitHub**

## Deployment

The application is deployed using **Streamlit Community Cloud**.

Python **3.12** is used for the deployment environment.

The deployed application loads the saved TF-IDF vectorizer, Linear SVC model,
and evaluation artifacts directly from the repository.

## Project Structure

```text
Fake-News-Detection/
|-- app.py
|-- execute_pipeline.py
|-- phase2_features.py
|-- phase3_errors.py
|-- final_model.pkl
|-- final_vectorizer.pkl
|-- evaluation_results.json
|-- requirements.txt
|-- runtime.txt
|-- test_app.py
|-- README.md
`-- .gitignore