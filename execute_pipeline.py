#!/usr/bin/env python3
"""
FAKE NEWS DETECTION - COMPLETE PIPELINE EXECUTION
Runs all ML steps with proper methodology validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import json
import os
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.metrics import make_scorer

import warnings
warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("FAKE NEWS DETECTION - COMPLETE PIPELINE EXECUTION")
print("="*80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[STEP 1] LOADING DATASET")
print("-" * 80)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get('FAKE_NEWS_DATA_PATH', PROJECT_DIR / 'FakeNewsNet.csv'))
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"✗ ERROR: Dataset not found at {DATA_PATH}")
    print("  Set FAKE_NEWS_DATA_PATH or place FakeNewsNet.csv beside this script.")
    exit(1)

# ============================================================================
# 2. DATA QUALITY CHECK
# ============================================================================
print("\n[STEP 2] DATA QUALITY ASSESSMENT")
print("-" * 80)

print(f"Missing values:")
print(df.isnull().sum())
dup_count = df['title'].duplicated().sum()
print(f"Duplicate titles: {dup_count} ({100*dup_count/len(df):.2f}%)")

# ============================================================================
# 3. CLASS DISTRIBUTION
# ============================================================================
print("\n[STEP 3] CLASS DISTRIBUTION ANALYSIS")
print("-" * 80)

class_labels = {1: 'Real', 0: 'Fake'}
class_counts = df['real'].value_counts().sort_index()

print("CLASS DISTRIBUTION:")
for cls in sorted(class_counts.index):
    count = class_counts[cls]
    pct = 100 * count / len(df)
    print(f"  {class_labels[cls]:6s}: {count:6d} samples ({pct:5.2f}%)")

imbalance_ratio = class_counts[1] / class_counts[0]
baseline_acc = class_counts[1] / len(df)

print(f"\nImbalance Ratio (Real:Fake): {imbalance_ratio:.2f}:1")
print(f"Majority Class Baseline Accuracy: {baseline_acc:.4f}")

# Baseline prediction (always predict majority class)
y_baseline_pred = np.ones(len(df))
baseline_f1_fake = f1_score(df['real'], y_baseline_pred, pos_label=0, zero_division=0)
print(f"Baseline F1 (Fake class): {baseline_f1_fake:.4f}")

# ============================================================================
# 4. TEXT PREPROCESSING
# ============================================================================
print("\n[STEP 4] TEXT PREPROCESSING")
print("-" * 80)

def clean_text(text):
    """Clean news title"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['title_clean'] = df['title'].apply(clean_text)
print(f"✓ Text cleaning completed")
print(f"  Sample cleaned text:")
print(f"    Original: {df.iloc[0]['title'][:60]}")
print(f"    Cleaned:  {df.iloc[0]['title_clean'][:60]}")

# ============================================================================
# 5. TRAIN/TEST SPLIT (CRITICAL: BEFORE VECTORIZATION)
# ============================================================================
print("\n[STEP 5] TRAIN/TEST SPLIT")
print("-" * 80)

X = df['title_clean'].values
y = df['real'].values

# CRITICAL: Split BEFORE vectorization
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"✓ Train/test split completed")
print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples:     {len(X_test):,}")
print(f"  Training set class distribution:")
for cls in sorted(np.unique(y_train)):
    count = (y_train == cls).sum()
    pct = 100 * count / len(y_train)
    print(f"    {class_labels[cls]:6s}: {count:6d} ({pct:5.2f}%)")
print(f"  Test set class distribution:")
for cls in sorted(np.unique(y_test)):
    count = (y_test == cls).sum()
    pct = 100 * count / len(y_test)
    print(f"    {class_labels[cls]:6s}: {count:6d} ({pct:5.2f}%)")

# ============================================================================
# 6. TF-IDF VECTORIZATION (NO LEAKAGE)
# ============================================================================
print("\n[STEP 6] TF-IDF VECTORIZATION")
print("-" * 80)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

# CRITICAL: Fit ONLY on training data
X_train_tfidf = vectorizer.fit_transform(X_train)
# Transform test data using fitted vectorizer (no refit)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ TF-IDF vectorization completed (NO LEAKAGE)")
print(f"  Features extracted: {X_train_tfidf.shape[1]:,}")
print(f"  Training TF-IDF shape: {X_train_tfidf.shape}")
print(f"  Test TF-IDF shape: {X_test_tfidf.shape}")

# ============================================================================
# 7. MODEL TRAINING & EVALUATION FUNCTION
# ============================================================================
print("\n[STEP 7] TRAINING MODELS")
print("-" * 80)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate model with comprehensive metrics"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Probability scores for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None
    
    results = {
        'model_name': model_name,
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision_real': precision_score(y_test, y_pred, pos_label=1),
        'recall_real': recall_score(y_test, y_pred, pos_label=1),
        'f1_real': f1_score(y_test, y_pred, pos_label=1),
        'precision_fake': precision_score(y_test, y_pred, pos_label=0),
        'recall_fake': recall_score(y_test, y_pred, pos_label=0),
        'f1_fake': f1_score(y_test, y_pred, pos_label=0),
    }
    
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_proba)
    else:
        results['roc_auc'] = None
    
    return results

# Train models
print("Training Logistic Regression...", end='')
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr_results = evaluate_model(lr, X_train_tfidf, X_test_tfidf, y_train, y_test, "Logistic Regression")
print(" [OK]")

print("Training Multinomial Naive Bayes...", end='')
nb = MultinomialNB()
nb_results = evaluate_model(nb, X_train_tfidf, X_test_tfidf, y_train, y_test, "Multinomial Naive Bayes")
print(" [OK]")

print("Training Linear SVC...", end='')
svm = LinearSVC(max_iter=2000, random_state=RANDOM_STATE, dual=False)
svm_results = evaluate_model(svm, X_train_tfidf, X_test_tfidf, y_train, y_test, "Linear SVC")
print(" [OK]")

model_results = [lr_results, nb_results, svm_results]

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================
print("\n[STEP 8] MODEL COMPARISON")
print("-" * 80)

print(f"\n{'Model':<25} {'Accuracy':>10} {'Bal.Acc':>10} {'Real F1':>10} {'Fake F1':>10} {'Real Rec':>10} {'Fake Rec':>10} {'ROC-AUC':>10}")
print("="*120)
for result in model_results:
    roc_auc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] else "N/A"
    print(f"{result['model_name']:<25} {result['accuracy']:>10.4f} {result['balanced_accuracy']:>10.4f} {result['f1_real']:>10.4f} {result['f1_fake']:>10.4f} {result['recall_real']:>10.4f} {result['recall_fake']:>10.4f} {roc_auc_str:>10}")

# ============================================================================
# 9. DETAILED PER-MODEL EVALUATION
# ============================================================================
print("\n[STEP 9] DETAILED PER-MODEL EVALUATION")
print("-" * 80)

for result in model_results:
    print(f"\nMODEL: {result['model_name']}")
    print(f"{'='*70}")
    
    print(f"Overall Metrics:")
    print(f"  Accuracy:           {result['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {result['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC:            {result['roc_auc']:.4f}" if result['roc_auc'] else "")
    
    print(f"\nREAL NEWS (Majority Class):")
    print(f"  Precision: {result['precision_real']:.4f}")
    print(f"  Recall:    {result['recall_real']:.4f}")
    print(f"  F1-Score:  {result['f1_real']:.4f}")
    
    print(f"\nFAKE NEWS (Minority Class) - PRIMARY METRIC:")
    print(f"  Precision: {result['precision_fake']:.4f}")
    print(f"  Recall:    {result['recall_fake']:.4f}")
    print(f"  F1-Score:  {result['f1_fake']:.4f}")
    
    cm = confusion_matrix(y_test, result['y_pred'])
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

# ============================================================================
# 10. CROSS-VALIDATION
# ============================================================================
print("\n[STEP 10] CROSS-VALIDATION (5-FOLD STRATIFIED)")
print("-" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = []
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'f1_fake': make_scorer(f1_score, pos_label=0, zero_division=0),
    'balanced_accuracy': 'balanced_accuracy'
}

for model_result in model_results:
    model_class = model_result['model'].__class__
    model_name = model_result['model_name']
    
    # Recreate fresh model
    if model_class == LogisticRegression:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_class == MultinomialNB:
        model = MultinomialNB()
    else:
        model = LinearSVC(max_iter=2000, random_state=RANDOM_STATE, dual=False)
    
    cv_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2), max_features=15000, min_df=2,
            max_df=0.95, stop_words='english'
        )),
        ('classifier', model)
    ])

    cv_scores = cross_validate(
        cv_pipeline, X_train, y_train,
        cv=skf, scoring=scoring, n_jobs=-1
    )

    cv_summary = {
        'model_name': model_name,
        'accuracy_mean': float(cv_scores['test_accuracy'].mean()),
        'accuracy_std': float(cv_scores['test_accuracy'].std()),
        'f1_macro_mean': float(cv_scores['test_f1_macro'].mean()),
        'f1_macro_std': float(cv_scores['test_f1_macro'].std()),
        'f1_fake_mean': float(cv_scores['test_f1_fake'].mean()),
        'f1_fake_std': float(cv_scores['test_f1_fake'].std()),
        'balanced_accuracy_mean': float(cv_scores['test_balanced_accuracy'].mean()),
        'balanced_accuracy_std': float(cv_scores['test_balanced_accuracy'].std())
    }
    cv_results.append(cv_summary)
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:         {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
    print(f"  Macro F1:          {cv_summary['f1_macro_mean']:.4f} ± {cv_summary['f1_macro_std']:.4f}")
    print(f"  Fake F1 (label 0): {cv_summary['f1_fake_mean']:.4f} ± {cv_summary['f1_fake_std']:.4f}")
    print(f"  Balanced Accuracy: {cv_scores['test_balanced_accuracy'].mean():.4f} ± {cv_scores['test_balanced_accuracy'].std():.4f}")

# ============================================================================
# 11. SELECT BEST MODEL
# ============================================================================
print("\n[STEP 11] MODEL SELECTION")
print("-" * 80)

# Select using training-only CV Fake F1; the held-out test set is not used.
best_idx = int(np.argmax([r['f1_fake_mean'] for r in cv_results]))
best_result = model_results[best_idx]
selected_model = best_result['model']
selected_name = best_result['model_name']
selected_cv = cv_results[best_idx]

print(f"\nSELECTED MODEL: {selected_name}")
print(f"Selection Criterion: Highest 5-fold CV Fake-class F1 on training data only")
print(f"  Fake F1-Score: {best_result['f1_fake']:.4f}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")

# ============================================================================
# 12. ERROR ANALYSIS
# ============================================================================
print("\n[STEP 12] ERROR ANALYSIS")
print("-" * 80)

y_pred_final = best_result['y_pred']
cm_final = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm_final[0,0], cm_final[0,1], cm_final[1,0], cm_final[1,1]

fake_false_negatives = (y_test == 0) & (y_pred_final == 1)
fake_false_positives = (y_test == 1) & (y_pred_final == 0)
correct = y_test == y_pred_final

print(f"\nError Breakdown:")
print(f"  Correct predictions:     {correct.sum():5d} ({100*correct.sum()/len(y_test):5.1f}%)")
print(f"  Incorrect predictions:   {(~correct).sum():5d} ({100*(~correct).sum()/len(y_test):5.1f}%)")
print(f"\n  Fake-class false negatives (Fake->Real): {fake_false_negatives.sum():5d} ({100*fake_false_negatives.sum()/len(y_test):5.1f}%)")
print(f"    [Risky: Fake news passes as real]")
print(f"\n  Fake-class false positives (Real->Fake): {fake_false_positives.sum():5d} ({100*fake_false_positives.sum()/len(y_test):5.1f}%)")
print(f"    [False alarm: Real news flagged as fake]")

# ============================================================================
# 13. SAVE MACHINE-READABLE EVALUATION ARTIFACT
# ============================================================================
print("\n[STEP 13] SAVING EVALUATION RESULTS")
print("-" * 80)

evaluation_path = PROJECT_DIR / 'evaluation_results.json'
test_precision_macro = precision_score(y_test, y_pred_final, average='macro', zero_division=0)
test_recall_macro = recall_score(y_test, y_pred_final, average='macro', zero_division=0)
test_f1_macro = f1_score(y_test, y_pred_final, average='macro', zero_division=0)
test_precision_weighted = precision_score(y_test, y_pred_final, average='weighted', zero_division=0)
test_recall_weighted = recall_score(y_test, y_pred_final, average='weighted', zero_division=0)
test_f1_weighted = f1_score(y_test, y_pred_final, average='weighted', zero_division=0)

model_summaries = []
for result in model_results:
    model_summaries.append({
        'model_name': result['model_name'],
        'accuracy': float(result['accuracy']),
        'balanced_accuracy': float(result['balanced_accuracy']),
        'roc_auc': float(result['roc_auc']) if result['roc_auc'] is not None else None,
        'precision_real': float(result['precision_real']),
        'recall_real': float(result['recall_real']),
        'f1_real': float(result['f1_real']),
        'precision_fake': float(result['precision_fake']),
        'recall_fake': float(result['recall_fake']),
        'f1_fake': float(result['f1_fake'])
    })

evaluation_results = {
    'dataset': {
        'name': 'FakeNewsNet',
        'total_samples': int(len(df)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_test_split': {'train_fraction': 0.8, 'test_fraction': 0.2},
        'class_counts': {'fake': int((y == 0).sum()), 'real': int((y == 1).sum())}
    },
    'baseline': {
        'strategy': 'always predict Real',
        'accuracy': float(baseline_acc),
        'fake_f1': float(baseline_f1_fake),
        'confusion_matrix': confusion_matrix(y_test, np.ones(len(y_test), dtype=int)).tolist()
    },
    'selection': {
        'criterion': 'highest cross-validation Fake-class F1 on training data only',
        'selected_model': selected_name
    },
    'selected_model': {
        'model_name': selected_name,
        'test': {
            'accuracy': float(best_result['accuracy']),
            'precision_macro': float(test_precision_macro),
            'recall_macro': float(test_recall_macro),
            'f1_macro': float(test_f1_macro),
            'precision_weighted': float(test_precision_weighted),
            'recall_weighted': float(test_recall_weighted),
            'f1_weighted': float(test_f1_weighted),
            'balanced_accuracy': float(best_result['balanced_accuracy']),
            'roc_auc': float(best_result['roc_auc']) if best_result['roc_auc'] is not None else None,
            'class_metrics': {
                'fake': {
                    'precision': float(best_result['precision_fake']),
                    'recall': float(best_result['recall_fake']),
                    'f1': float(best_result['f1_fake'])
                },
                'real': {
                    'precision': float(best_result['precision_real']),
                    'recall': float(best_result['recall_real']),
                    'f1': float(best_result['f1_real'])
                }
            },
            'confusion_matrix': cm_final.tolist(),
            'confusion_matrix_labels': ['Fake', 'Real']
        },
        'cross_validation': selected_cv
    },
    'models': model_summaries,
    'cross_validation': cv_results
}

with open(evaluation_path, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2)
print(f"✓ Evaluation results saved: {evaluation_path}")

# ============================================================================
# 14. SAVE MODEL AND VECTORIZER
# ============================================================================
print("\n[STEP 14] SAVING MODEL AND VECTORIZER")
print("-" * 80)

MODEL_PATH = PROJECT_DIR / 'final_model.pkl'
VECTORIZER_PATH = PROJECT_DIR / 'final_vectorizer.pkl'

try:
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(selected_model, f)
    print(f"✓ Model saved: {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error saving model: {e}")

try:
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved: {VECTORIZER_PATH}")
except Exception as e:
    print(f"✗ Error saving vectorizer: {e}")

# ============================================================================
# 15. VERIFY SAVED MODEL CAN LOAD AND PREDICT
# ============================================================================
print("\n[STEP 15] VERIFICATION - LOAD AND PREDICT")
print("-" * 80)

try:
    # Load saved model and vectorizer
    with open(MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
    print(f"✓ Vectorizer loaded successfully from: {VECTORIZER_PATH}")
    
    # Test prediction on a sample
    test_title = "Breaking news: Major discovery announced"
    test_cleaned = clean_text(test_title)
    test_vector = loaded_vectorizer.transform([test_cleaned])
    test_pred = loaded_model.predict(test_vector)
    
    if hasattr(loaded_model, 'decision_function'):
        test_confidence = loaded_model.decision_function(test_vector)[0]
    else:
        test_confidence = None
    
    print(f"\n✓ Test prediction successful:")
    print(f"  Input: '{test_title}'")
    print(f"  Cleaned: '{test_cleaned}'")
    print(f"  Prediction: {'Real' if test_pred[0]==1 else 'Fake'}")
    if test_confidence is not None:
        print(f"  Confidence: {test_confidence:.4f}")
    
except Exception as e:
    print(f"✗ Error loading/predicting: {e}")

# ============================================================================
# 16. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL EXECUTION SUMMARY")
print("="*80)

print(f"\nDATASET:")
print(f"  Total samples: {len(df):,}")
print(f"  Train samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Real news: {sum(y==1):,} (75.2%)")
print(f"  Fake news: {sum(y==0):,} (24.8%)")

print(f"\nBASELINE (Majority Class Classifier):")
print(f"  Accuracy: {baseline_acc:.4f}")
print(f"  F1 (Fake): {baseline_f1_fake:.4f}")

print(f"\nBEST MODEL: {selected_name}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}" if best_result['roc_auc'] else "")
print(f"\n  Fake News Detection (PRIMARY METRIC):")
print(f"    Precision: {best_result['precision_fake']:.4f}")
print(f"    Recall:    {best_result['recall_fake']:.4f}")
print(f"    F1-Score:  {best_result['f1_fake']:.4f}")

print(f"\n  Real News Detection:")
print(f"    Precision: {best_result['precision_real']:.4f}")
print(f"    Recall:    {best_result['recall_real']:.4f}")
print(f"    F1-Score:  {best_result['f1_real']:.4f}")

print(f"\nSAVED FILES:")
print(f"  Model: {MODEL_PATH}")
print(f"  Vectorizer: {VECTORIZER_PATH}")

print(f"\nSTATUS: ✓ PIPELINE EXECUTION COMPLETE")
print("="*80 + "\n")
