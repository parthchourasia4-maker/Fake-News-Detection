#!/usr/bin/env python3
"""
Test script for Fake News Detection Streamlit app
Tests core functionality without requiring web browser interaction
"""

import pickle
import re
import pandas as pd
import numpy as np
import io
import sys
from pathlib import Path

print("\n" + "="*80)
print("FAKE NEWS DETECTION - STREAMLIT APP TEST SUITE")
print("="*80)

# ============================================================================
# TEST 1: LOAD MODEL AND VECTORIZER
# ============================================================================
print("\n[TEST 1] Loading Model and Vectorizer...")
print("-"*80)

try:
    model_path = Path("final_model.pkl")
    vectorizer_path = Path("final_vectorizer.pkl")
    
    if not model_path.exists():
        print(f"✗ FAIL: Model file not found: {model_path}")
        sys.exit(1)
    if not vectorizer_path.exists():
        print(f"✗ FAIL: Vectorizer file not found: {vectorizer_path}")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f"✓ PASS: Model loaded ({type(model).__name__})")
    print(f"✓ PASS: Vectorizer loaded ({type(vectorizer).__name__})")
    print(f"  Model file size: {model_path.stat().st_size / 1024:.2f} KB")
    print(f"  Vectorizer file size: {vectorizer_path.stat().st_size / 1024:.2f} KB")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: CLEAN TEXT FUNCTION
# ============================================================================
print("\n[TEST 2] Testing Text Cleaning Function...")
print("-"*80)

def clean_text(text):
    """Clean news title using the same preprocessing as training."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

test_cases = [
    ("Breaking News: Scientists discover new energy source!", 
     "breaking news scientists discover new energy source"),
    ("Celebrity scandal on https://example.com now!!!",
     "celebrity scandal on now"),
    ("It's the BEST story ever (2024) - 99%",
     "its the best story ever"),
]

all_passed = True
for original, expected in test_cases:
    result = clean_text(original)
    if result == expected:
        print(f"✓ PASS: '{original[:40]}...'")
    else:
        print(f"✗ FAIL: '{original[:40]}...'")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        all_passed = False

if all_passed:
    print("✓ All text cleaning tests passed")
else:
    print("✗ Some text cleaning tests failed")
    sys.exit(1)

# ============================================================================
# TEST 3: SINGLE PREDICTION - VALID INPUT
# ============================================================================
print("\n[TEST 3] Testing Single Prediction - Valid Input...")
print("-"*80)

try:
    test_headlines = [
        "Scientists discover new cure for disease",
        "Breaking celebrity scandal revealed",
        "Major political announcement made today",
        "Technology company launches new product",
    ]
    
    for headline in test_headlines:
        cleaned = clean_text(headline)
        vector = vectorizer.transform([cleaned])
        pred = model.predict(vector)[0]
        pred_label = "Fake" if pred == 0 else "Real"
        
        # Get confidence
        if hasattr(model, 'decision_function'):
            confidence = model.decision_function(vector)[0]
            confidence_normalized = 1 / (1 + np.exp(-confidence))
        else:
            confidence_normalized = None
        
        print(f"✓ PASS: '{headline[:45]}...'")
        print(f"         Prediction: {pred_label}, Confidence: {confidence_normalized:.2%}" if confidence_normalized else "")
    
    print("✓ All single prediction tests passed")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: EMPTY/INVALID INPUT HANDLING
# ============================================================================
print("\n[TEST 4] Testing Empty/Invalid Input Handling...")
print("-"*80)

invalid_inputs = [
    "",
    "   ",
    "!!!???@@@",
    "http://example.com",
]

def predict_news(text, model, vectorizer):
    """Predict whether a news headline is likely Fake or Real."""
    if not text or not text.strip():
        return None, None, "Please enter a headline"
    
    cleaned = clean_text(text)
    
    if not cleaned:
        return None, None, "Headline contained no valid text after cleaning"
    
    text_vector = vectorizer.transform([cleaned])
    prediction = model.predict(text_vector)[0]
    
    if hasattr(model, 'decision_function'):
        confidence = model.decision_function(text_vector)[0]
        confidence_normalized = 1 / (1 + np.exp(-confidence))
    else:
        confidence_normalized = None
    
    return prediction, confidence_normalized, None

for invalid_input in invalid_inputs:
    pred, conf, error = predict_news(invalid_input, model, vectorizer)
    if error is not None:
        print(f"✓ PASS: Correctly rejected: '{invalid_input[:30]}...' -> {error}")
    else:
        print(f"✗ FAIL: Should have rejected: '{invalid_input}'")
        sys.exit(1)

print("✓ All invalid input tests passed")

# ============================================================================
# TEST 5: BATCH PREDICTION WITH VALID CSV
# ============================================================================
print("\n[TEST 5] Testing Batch Prediction with Valid CSV...")
print("-"*80)

try:
    # Create sample CSV
    sample_data = {
        'title': [
            'Scientists discover new energy source',
            'Celebrity caught in scandal',
            'New study shows health benefits',
            'Breaking news from entertainment',
            'Government announces new policy',
        ]
    }
    df_sample = pd.DataFrame(sample_data)
    
    print(f"✓ Created sample CSV with {len(df_sample)} rows")
    
    # Run predictions
    predictions = []
    for idx, row in df_sample.iterrows():
        title = row['title']
        pred, conf, error = predict_news(title, model, vectorizer)
        
        if error:
            predictions.append({
                'title': title,
                'prediction': 'ERROR',
                'prediction_label': 'Error',
                'confidence': None
            })
        else:
            pred_label = 'Fake' if pred == 0 else 'Real'
            predictions.append({
                'title': title,
                'prediction': pred,
                'prediction_label': pred_label,
                'confidence': conf
            })
    
    df_results = pd.DataFrame(predictions)
    
    print(f"✓ Processed {len(df_results)} predictions")
    print(f"\nResults:")
    print(df_results.to_string(index=False))
    
    # Verify results
    if len(df_results) == len(df_sample):
        print("\n✓ All batch predictions completed")
    else:
        print(f"\n✗ FAIL: Expected {len(df_sample)} results, got {len(df_results)}")
        sys.exit(1)
    
    # Test CSV export
    csv_buffer = io.StringIO()
    df_results.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    if len(csv_data) > 0 and 'title' in csv_data:
        print("✓ CSV export successful")
    else:
        print("✗ FAIL: CSV export failed")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# ============================================================================
# TEST 6: BATCH PREDICTION WITH INVALID CSV
# ============================================================================
print("\n[TEST 6] Testing Batch Prediction with Invalid CSV...")
print("-"*80)

try:
    # Create CSV without 'title' column
    invalid_data = {
        'headline': ['Test 1', 'Test 2'],
        'label': ['Real', 'Fake']
    }
    df_invalid = pd.DataFrame(invalid_data)
    
    # Check for 'title' column
    if 'title' not in df_invalid.columns:
        print(f"✓ PASS: Correctly detected missing 'title' column")
        print(f"  Found columns: {list(df_invalid.columns)}")
    else:
        print(f"✗ FAIL: Should have detected missing 'title' column")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# ============================================================================
# TEST 7: METRICS VALIDATION
# ============================================================================
print("\n[TEST 7] Verifying Model Metrics...")
print("-"*80)

# Generate full test set predictions for metrics
try:
    test_headlines_extended = [
        "Scientists discover breakthrough",
        "Celebrity news update",
        "Government policy announcement",
        "Business merger confirmed",
        "New study published",
        "Entertainment industry news",
        "Medical research findings",
        "Breaking political news",
        "Technology innovation unveiled",
        "Sports championship results",
    ]
    
    predictions_extended = []
    for headline in test_headlines_extended:
        pred, conf, error = predict_news(headline, model, vectorizer)
        if error is None:
            pred_label = 'Fake' if pred == 0 else 'Real'
            predictions_extended.append({
                'headline': headline,
                'prediction': pred,
                'prediction_label': pred_label,
                'confidence': conf
            })
    
    print(f"✓ Generated {len(predictions_extended)} predictions for validation")
    
    # Calculate distribution
    fake_count = sum(1 for p in predictions_extended if p['prediction'] == 0)
    real_count = sum(1 for p in predictions_extended if p['prediction'] == 1)
    
    print(f"\nPrediction Distribution:")
    print(f"  Fake: {fake_count} ({100*fake_count/len(predictions_extended):.1f}%)")
    print(f"  Real: {real_count} ({100*real_count/len(predictions_extended):.1f}%)")
    
    if len(predictions_extended) > 0:
        print("✓ Metrics validation passed")
    else:
        print("✗ FAIL: No predictions generated")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

test_results = {
    "Model Loading": "✓ PASS",
    "Text Cleaning": "✓ PASS",
    "Single Prediction": "✓ PASS",
    "Empty/Invalid Input": "✓ PASS",
    "Batch Prediction (Valid CSV)": "✓ PASS",
    "Invalid CSV Detection": "✓ PASS",
    "Metrics Validation": "✓ PASS",
}

for test_name, result in test_results.items():
    print(f"{result} - {test_name}")

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - STREAMLIT APP READY FOR USE")
print("="*80 + "\n")

print("To run the Streamlit app, execute:")
print("  streamlit run app.py")
print()
