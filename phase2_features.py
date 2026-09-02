"""
PHASE 2: Feature Engineering and Model Optimization
Testing different TF-IDF configurations for performance improvement
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, 
    precision_score, recall_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*100)
print("PHASE 2: FEATURE ENGINEERING & OPTIMIZATION")
print("="*100)

# Load and prepare data
DATA_PATH = Path(os.environ.get('FAKE_NEWS_DATA_PATH', Path(__file__).resolve().parent / 'FakeNewsNet.csv'))
df = pd.read_csv(DATA_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['title_clean'] = df['title'].apply(clean_text)
X = df['title_clean'].values
y = df['real'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTest set class distribution:")
print(f"  Real: {sum(y_test==1):5d} ({100*sum(y_test==1)/len(y_test):5.1f}%)")
print(f"  Fake: {sum(y_test==0):5d} ({100*sum(y_test==0)/len(y_test):5.1f}%)")

# Define feature configurations to test
configs = [
    {
        'name': 'Baseline (Unigrams only)',
        'ngram_range': (1, 1),
        'max_features': 10000,
        'min_df': 2,
        'max_df': 0.95
    },
    {
        'name': 'Bigrams added',
        'ngram_range': (1, 2),
        'max_features': 10000,
        'min_df': 2,
        'max_df': 0.95
    },
    {
        'name': 'Trigrams added',
        'ngram_range': (1, 3),
        'max_features': 10000,
        'min_df': 2,
        'max_df': 0.95
    },
    {
        'name': 'More features (15k)',
        'ngram_range': (1, 1),
        'max_features': 15000,
        'min_df': 2,
        'max_df': 0.95
    },
    {
        'name': 'Fewer features (5k)',
        'ngram_range': (1, 1),
        'max_features': 5000,
        'min_df': 2,
        'max_df': 0.95
    },
    {
        'name': 'Higher max_df (0.99)',
        'ngram_range': (1, 1),
        'max_features': 10000,
        'min_df': 2,
        'max_df': 0.99
    },
    {
        'name': 'Bigrams + more features (15k)',
        'ngram_range': (1, 2),
        'max_features': 15000,
        'min_df': 2,
        'max_df': 0.95
    }
]

print(f"\n" + "="*100)
print(f"TESTING {len(configs)} FEATURE CONFIGURATIONS")
print("="*100)

results = []

for config in configs:
    print(f"\nTesting: {config['name']}")
    print(f"  ngram_range: {config['ngram_range']}")
    print(f"  max_features: {config['max_features']}")
    print(f"  min_df: {config['min_df']}, max_df: {config['max_df']}")
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=config['ngram_range'],
        max_features=config['max_features'],
        min_df=config['min_df'],
        max_df=config['max_df'],
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Linear SVC (best performing model from Phase 1)
    model = LinearSVC(max_iter=2000, random_state=RANDOM_STATE, dual=False)
    model.fit(X_train_tfidf, y_train)
    
    # Predict
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.decision_function(X_test_tfidf)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_fake = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_real = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_fake = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision_fake = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    n_features = X_train_tfidf.shape[1]
    
    results.append({
        'config': config['name'],
        'n_features': n_features,
        'accuracy': accuracy,
        'balanced_acc': balanced_acc,
        'f1_fake': f1_fake,
        'f1_real': f1_real,
        'recall_fake': recall_fake,
        'precision_fake': precision_fake,
        'roc_auc': roc_auc,
        'vectorizer': vectorizer,
        'model': model
    })
    
    print(f"  ✓ Features extracted: {n_features}")
    print(f"    Accuracy: {accuracy:.4f} | Balanced Acc: {balanced_acc:.4f}")
    print(f"    Fake F1: {f1_fake:.4f} | Real F1: {f1_real:.4f}")
    print(f"    Fake Recall: {recall_fake:.4f} | Fake Precision: {precision_fake:.4f}")
    print(f"    ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print(f"\n" + "="*100)
print(f"COMPARISON TABLE")
print("="*100)
print(f"{'Config':<35} {'N Features':>12} {'Accuracy':>10} {'Bal.Acc':>10} {'Fake F1':>10} {'Real F1':>10} {'ROC-AUC':>10}")
print("="*100)

for result in results:
    print(f"{result['config']:<35} {result['n_features']:>12} {result['accuracy']:>10.4f} {result['balanced_acc']:>10.4f} {result['f1_fake']:>10.4f} {result['f1_real']:>10.4f} {result['roc_auc']:>10.4f}")

print("="*100)

# Find best configuration
best_idx = np.argmax([r['f1_fake'] for r in results])
best_result = results[best_idx]

print(f"\n✓ BEST CONFIGURATION: {best_result['config']}")
print(f"  F1-Score (Fake class): {best_result['f1_fake']:.4f}")
print(f"  F1-Score (Real class): {best_result['f1_real']:.4f}")
print(f"  Balanced Accuracy: {best_result['balanced_acc']:.4f}")
print(f"  Accuracy: {best_result['accuracy']:.4f}")
print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"  Features: {best_result['n_features']}")

# ============================================================================
# COMPARISON TO BASELINE
# ============================================================================
baseline_result = results[0]  # Unigrams only
improvement_fake_f1 = best_result['f1_fake'] - baseline_result['f1_fake']
improvement_acc = best_result['accuracy'] - baseline_result['accuracy']

print(f"\nImprovement over Baseline (Unigrams only):")
print(f"  Fake F1: {baseline_result['f1_fake']:.4f} → {best_result['f1_fake']:.4f} ({improvement_fake_f1:+.4f})")
print(f"  Accuracy: {baseline_result['accuracy']:.4f} → {best_result['accuracy']:.4f} ({improvement_acc:+.4f})")

# ============================================================================
# SAVE BEST MODEL INFO
# ============================================================================
print(f"\n" + "="*100)
print(f"SELECTED FOR NEXT PHASE: {best_result['config']}")
print("="*100)

print(f"\nPHASE 2 COMPLETE: Best performing configuration identified")
