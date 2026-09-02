"""
PHASE 3: Comprehensive Error Analysis
Understand what the model gets wrong and why
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*100)
print("PHASE 3: ERROR ANALYSIS")
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

indices = np.arange(len(X))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Preserve the shuffled test-row alignment for error examples.
df_test = df.iloc[test_indices].reset_index(drop=True)

# Use best configuration from Phase 2
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LinearSVC(max_iter=2000, random_state=RANDOM_STATE, dual=False)
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)
y_proba = model.decision_function(X_test_tfidf)

# Identify error types
cm = confusion_matrix(y_test, y_pred)
fake_true_positives = cm[0, 0]
fake_false_negatives = cm[0, 1]  # Predicted Real, actually Fake
fake_false_positives = cm[1, 0]  # Predicted Fake, actually Real
fake_true_negatives = cm[1, 1]

# Index arrays for each error type
fake_miss_mask = (y_test == 0) & (y_pred == 1)
fake_false_alarm_mask = (y_test == 1) & (y_pred == 0)
tp_mask = (y_test == 1) & (y_pred == 1)
tn_mask = (y_test == 0) & (y_pred == 0)

print(f"\nERROR BREAKDOWN:")
print(f"  Fake true positives (Correctly predicted Fake): {fake_true_positives:5d} ({100*fake_true_positives/len(y_test):5.1f}%)")
print(f"  Fake false negatives (Fake->Real):              {fake_false_negatives:5d} ({100*fake_false_negatives/len(y_test):5.1f}%) [MISSED]")
print(f"  Fake false positives (Real->Fake):              {fake_false_positives:5d} ({100*fake_false_positives/len(y_test):5.1f}%) [FALSE ALARM]")
print(f"  Fake true negatives (Correctly predicted Real): {fake_true_negatives:5d} ({100*fake_true_negatives/len(y_test):5.1f}%)")
print(f"  {'─'*70}")
print(f"  Correct predictions:                              {fake_true_positives + fake_true_negatives:5d} ({100*(fake_true_positives + fake_true_negatives)/len(y_test):5.1f}%)")
print(f"  Incorrect predictions:                            {fake_false_negatives + fake_false_positives:5d} ({100*(fake_false_negatives + fake_false_positives)/len(y_test):5.1f}%)")

# ============================================================================
# FALSE POSITIVES ANALYSIS (Fake predicted as Real)
# ============================================================================
print(f"\n" + "="*100)
print(f"FAKE-CLASS FALSE NEGATIVES: Fake news mislabeled as Real (Most Dangerous)")
print("="*100)

fp_indices = np.where(fake_miss_mask)[0]
if len(fp_indices) > 0:
    # Sort by confidence (model was most sure it was real)
    fp_confidences = y_proba[fp_indices]
    sorted_fp_idx = np.argsort(-fp_confidences)[:min(10, len(fp_indices))]
    
    print(f"\nShowing {min(10, len(fp_indices))} examples where fake news slipped through as real:")
    print(f"(Most confident wrong predictions)")
    print(f"\n{'─'*100}")
    
    for i, fp_idx in enumerate(sorted_fp_idx, 1):
        actual_idx = fp_indices[fp_idx]
        title = X_test[actual_idx]
        confidence = y_proba[actual_idx]
        print(f"\n{i}. {title[:80]}...")
        print(f"   Original: {df_test.iloc[actual_idx]['title'][:60]}...")
        print(f"   Prediction confidence (Real): {confidence:.4f}")
        print(f"   Model confident this is Real, but it's actually Fake")
    
    print(f"\n(Total false positives: {len(fp_indices)}, showing {min(10, len(fp_indices))})")
else:
    print("No false positives found.")

# ============================================================================
# FALSE NEGATIVES ANALYSIS (Real predicted as Fake)
# ============================================================================
print(f"\n" + "="*100)
print(f"FAKE-CLASS FALSE POSITIVES: Real news mislabeled as Fake (False Alarms)")
print("="*100)

fn_indices = np.where(fake_false_alarm_mask)[0]
if len(fn_indices) > 0:
    # Sort by confidence (model was most sure it was fake)
    fn_confidences = -y_proba[fn_indices]  # Negate because decision function is Real bias
    sorted_fn_idx = np.argsort(-fn_confidences)[:min(10, len(fn_indices))]
    
    print(f"\nShowing {min(10, len(fn_indices))} examples where real news was flagged as fake:")
    print(f"(Most confident wrong predictions)")
    print(f"\n{'─'*100}")
    
    for i, fn_idx in enumerate(sorted_fn_idx, 1):
        actual_idx = fn_indices[fn_idx]
        title = X_test[actual_idx]
        confidence = y_proba[actual_idx]
        print(f"\n{i}. {title[:80]}...")
        print(f"   Original: {df_test.iloc[actual_idx]['title'][:60]}...")
        print(f"   Prediction confidence (Real): {confidence:.4f} (negative = predicted Fake)")
        print(f"   Model confident this is Fake, but it's actually Real")
    
    print(f"\n(Total false negatives: {len(fn_indices)}, showing {min(10, len(fn_indices))})")
else:
    print("No false negatives found.")

# ============================================================================
# FEATURE IMPORTANCE FOR ERRORS
# ============================================================================
print(f"\n" + "="*100)
print(f"FEATURE ANALYSIS: Top discriminative terms")
print("="*100)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Top positive features (indicate Real)
top_real_idx = np.argsort(coefficients)[-15:]
print(f"\nTop 15 terms indicating REAL NEWS (Positive coefficients):")
for i, idx in enumerate(reversed(top_real_idx), 1):
    print(f"  {i:2d}. {feature_names[idx]:30s} : {coefficients[idx]:8.4f}")

# Top negative features (indicate Fake)
top_fake_idx = np.argsort(coefficients)[:15]
print(f"\nTop 15 terms indicating FAKE NEWS (Negative coefficients):")
for i, idx in enumerate(top_fake_idx, 1):
    print(f"  {i:2d}. {feature_names[idx]:30s} : {coefficients[idx]:8.4f}")

# ============================================================================
# PREDICTION CONFIDENCE DISTRIBUTION
# ============================================================================
print(f"\n" + "="*100)
print(f"PREDICTION CONFIDENCE ANALYSIS")
print("="*100)

print(f"\nDecision function score distribution:")
print(f"  Mean (all predictions): {np.mean(y_proba):.4f}")
print(f"  Std Dev: {np.std(y_proba):.4f}")
print(f"  Min: {np.min(y_proba):.4f} (most confident Fake prediction)")
print(f"  Max: {np.max(y_proba):.4f} (most confident Real prediction)")

# Predictions near decision boundary (uncertain)
uncertain_threshold = 0.5
uncertain_mask = np.abs(y_proba) < uncertain_threshold
uncertain_count = uncertain_mask.sum()
uncertain_pct = 100 * uncertain_count / len(y_pred)

print(f"\nUncertain predictions (|score| < {uncertain_threshold}):")
print(f"  Count: {uncertain_count} ({uncertain_pct:.1f}%)")
if uncertain_count > 0:
    uncertain_errors = uncertain_count - (y_pred[uncertain_mask] == y_test[uncertain_mask]).sum()
    print(f"  Errors in uncertain predictions: {uncertain_errors} ({100*uncertain_errors/uncertain_count:.1f}%)")

# ============================================================================
# ERROR PATTERNS
# ============================================================================
print(f"\n" + "="*100)
print(f"ERROR PATTERNS & INSIGHTS")
print("="*100)

print(f"\nFalse Positive Characteristics (Fake→Real):")
print(f"  Count: {len(fp_indices)}")
if len(fp_indices) > 0:
    fp_titles = [X_test[i] for i in fp_indices]
    avg_length = np.mean([len(t.split()) for t in fp_titles])
    print(f"  Average title length: {avg_length:.1f} words")
    print(f"  Characteristics: These fake titles may use language patterns similar to real news")

print(f"\nFalse Negative Characteristics (Real→Fake):")
print(f"  Count: {len(fn_indices)}")
if len(fn_indices) > 0:
    fn_titles = [X_test[i] for i in fn_indices]
    avg_length = np.mean([len(t.split()) for t in fn_titles])
    print(f"  Average title length: {avg_length:.1f} words")
    print(f"  Characteristics: These real titles may use language patterns similar to fake news")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print(f"\n" + "="*100)
print(f"KEY FINDINGS & RECOMMENDATIONS")
print("="*100)

print(f"""
1. MODEL BEHAVIOR:
    • Model correctly identifies {100*fake_true_negatives/(fake_true_negatives+fake_false_positives):.1f}% of Real news
    • Model correctly identifies {100*fake_true_positives/(fake_true_positives+fake_false_negatives):.1f}% of Fake news
    • Fake-class false-negative rate: {100*fake_false_negatives/(fake_true_positives+fake_false_negatives):.1f}% (risky - fake news passed as real)
   
2. DANGEROUS MISSES:
    • {fake_false_negatives} fake news articles pass through as real
    • These represent {100*fake_false_negatives/len(y_pred):.1f}% of test predictions
   • Pattern: Fake titles using language similar to legitimate sources
   
3. FALSE ALARMS:
    • {fake_false_positives} real news articles are flagged as fake
    • These represent {100*fake_false_positives/len(y_pred):.1f}% of test predictions
   • Pattern: Real titles using dramatic or unusual language

4. LIMITATION INSIGHTS:
   • Model uses title text only → loses important signals from:
     - Article body content
     - Source credibility history
     - Publication domain reputation
     - Temporal context and trending patterns
   
5. NEXT IMPROVEMENTS:
   • Add domain/source features (known fake sources)
   • Use article body text, not just title
   • Incorporate temporal information
   • Cross-check against fact-checking databases
   • Ensemble with domain-specific signals
   • Consider confidence thresholds (don't predict on uncertain samples)
""")

print(f"="*100)
print(f"PHASE 3 COMPLETE: Error patterns identified and analyzed")
print("="*100)
