import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    :root {
        --ink: #e7ecef;
        --muted: #9aa8ae;
        --canvas: #101719;
        --panel: #172124;
        --panel-raised: #1d2a2d;
        --line: #2b3b3e;
        --accent: #63c7b2;
        --accent-soft: #183936;
        --real: #86c8a4;
        --real-soft: #193329;
        --fake: #e39a91;
        --fake-soft: #382523;
        --warning: #d9b878;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #0d1416; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.15rem; }
    [data-testid="stSidebar"] hr { border-color: var(--line); margin: 1.4rem 0; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stSidebar"] .stRadio label { padding: .45rem .2rem; color: var(--muted); }
    [data-testid="stSidebar"] .stRadio label:hover { color: var(--ink); }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2.35rem !important; line-height: 1.1 !important; margin-bottom: .45rem !important; }
    h2 { font-size: 1.45rem !important; margin-top: 2rem !important; }
    h3 { font-size: 1.05rem !important; }
    p, li, label { color: #c5ced1; }
    .eyebrow { color: var(--accent); font-size: .74rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .55rem; }
    .page-intro { color: var(--muted); font-size: 1rem; max-width: 720px; margin-bottom: 1.6rem; }
    .brand-mark { color: var(--ink); font-size: 1.15rem; font-weight: 750; line-height: 1.15; }
    .brand-mark span { color: var(--accent); }
    .brand-caption { color: var(--muted); font-size: .78rem; margin-top: .35rem; }
    .section-rule { border-top: 1px solid var(--line); margin: 2rem 0 1.4rem; }
    .metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.05rem; min-height: 104px; }
    .metric-label { color: var(--muted); font-size: .77rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color: var(--ink); font-size: 1.65rem; font-weight: 700; margin-top: .45rem; }
    .metric-note { color: var(--muted); font-size: .76rem; margin-top: .2rem; }
    .insight-strip { display: flex; gap: .85rem; align-items: baseline; background: var(--accent-soft); border: 1px solid #2d5a53; border-left: 3px solid var(--accent); border-radius: 6px; padding: .85rem 1rem; margin: 1.25rem 0 1.55rem; }
    .insight-strip strong { color: var(--accent); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; white-space: nowrap; }
    .insight-strip span { color: var(--ink); font-size: .92rem; }
    .prediction-fake, .prediction-real { padding: 1.15rem 1.3rem; border-radius: 8px; margin: 1.2rem 0 .85rem; }
    .prediction-fake { background: var(--fake-soft); border: 1px solid #714b48; border-left: 4px solid var(--fake); }
    .prediction-real { background: var(--real-soft); border: 1px solid #3e6b55; border-left: 4px solid var(--real); }
    .prediction-fake h3, .prediction-real h3 { margin: 0 0 .35rem; }
    .prediction-fake h3 { color: var(--fake); }
    .prediction-real h3 { color: var(--real); }
    .confidence-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: .9rem 1rem; }
    .warning-box { background: #2c291f; border: 1px solid #5a5039; border-left: 3px solid var(--warning); border-radius: 6px; padding: 1rem 1.1rem; margin: 1.2rem 0; }
    .limitation-box { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.1rem; margin: .65rem 0; }
    .stButton > button { background: var(--accent); color: #10201e; border: 0; border-radius: 6px; font-weight: 700; min-height: 2.8rem; }
    .stButton > button:hover { background: #8bd9c5; color: #10201e; }
    .stTextArea textarea, .stTextInput input { background: #131c1e; color: var(--ink); border: 1px solid var(--line); border-radius: 6px; }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: .85rem 1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }
    .stInfo, .stSuccess, .stWarning, .stError { border-radius: 6px; }
    @media (max-width: 800px) { h1 { font-size: 1.85rem !important; } .page-intro { font-size: .95rem; } }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    try:
        model_path = Path(__file__).parent / "final_model.pkl"
        vectorizer_path = Path(__file__).parent / "final_vectorizer.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data
def load_evaluation_results():
    """Load metrics generated by the final evaluation pipeline."""
    try:
        results_path = Path(__file__).parent / "evaluation_results.json"
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def clean_text(text):
    """Clean news title using the same preprocessing as training."""
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text, model, vectorizer):
    """
    Predict whether a news headline is likely Fake or Real.
    Returns prediction (0=Fake, 1=Real) and confidence score.
    """
    if not text or not text.strip():
        return None, None, "Please enter a headline"
    
    # Clean the text
    cleaned = clean_text(text)
    
    if not cleaned:
        return None, None, "Headline contained no valid text after cleaning"
    
    # Vectorize
    text_vector = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(text_vector)[0]

    if hasattr(model, 'decision_function'):
        confidence = model.decision_function(text_vector)[0]
        confidence_normalized = 1 / (1 + np.exp(-confidence))
    else:
        confidence_normalized = None

    return prediction, confidence_normalized, None

def page_header(eyebrow, title, description):
    """Render the shared page hierarchy."""
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<div class="page-intro">{description}</div>', unsafe_allow_html=True)


def metric_card(label, value, note=""):
    """Render a compact metric card."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

# ============================================================================
# LOAD ARTIFACTS AND NAVIGATION
# ============================================================================
model, vectorizer, load_error = load_model_and_vectorizer()
evaluation, evaluation_error = load_evaluation_results()

if load_error:
    st.error(f"Error loading model: {load_error}")
    st.stop()
if evaluation_error:
    st.error(f"Error loading evaluation results: {evaluation_error}")
    st.stop()

selected_test = evaluation['selected_model']['test']
selected_cv = evaluation['selected_model']['cross_validation']

st.sidebar.markdown('<div class="brand-mark">Signal<span>/</span>Check</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-caption">Headline pattern analysis</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to:",
    options=[
        "🏠 Home & Prediction",
        "📊 Model Performance",
        "🔍 Feature Importance",
        "📥 Batch Prediction",
        "ℹ️ Project Information"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **About this project**

    A machine learning system trained to detect patterns in news headlines.

    **Dataset:** {evaluation['dataset']['name']} ({evaluation['dataset']['total_samples']:,} samples)

    **Model:** {evaluation['selection']['selected_model']} with TF-IDF features
    """
)

# ============================================================================
# PAGE 1: HOME & PREDICTION
# ============================================================================
if page == "🏠 Home & Prediction":
    page_header("NLP CLASSIFIER", "Fake News Detector", "A focused look at the language patterns a trained model associates with real and fake news headlines.")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.65, 1], gap="large")

    with col1:
        st.subheader("Analyze a news headline")
        headline = st.text_area("Headline", height=150, placeholder="Paste or type a headline here...", label_visibility="collapsed")
        if st.button("Analyze headline", width="stretch"):
            if not headline.strip():
                st.warning("⚠️ Please enter a headline to analyze.")
            else:
                prediction, confidence, error = predict_news(headline, model, vectorizer)
                if error:
                    st.error(f"Error: {error}")
                elif prediction == 0:
                    st.markdown('<div class="prediction-fake"><h3>Warning · Likely fake</h3>The model associates this headline with <b>fake news patterns</b> in its training data.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-real"><h3>Verified signal · Likely real</h3>The model associates this headline with <b>real news patterns</b> in its training data.</div>', unsafe_allow_html=True)
                if error is None:
                    st.markdown(f'<div class="confidence-card"><div class="metric-label">Model confidence</div><div class="metric-value">{confidence:.1%}</div><div class="metric-note">A decision-score-derived signal, not a calibrated probability.</div></div>', unsafe_allow_html=True)
                    with st.expander("Show preprocessing details"):
                        st.code(clean_text(headline), language=None)

    with col2:
        st.subheader("At a glance")
        metric_card("Test accuracy", f"{selected_test['accuracy']:.2%}", "Held-out test set")
        metric_card("Fake-class F1", f"{selected_test['class_metrics']['fake']['f1']:.4f}", "Primary selection metric")
        metric_card("Training samples", f"{evaluation['dataset']['train_samples']:,}", "FakeNewsNet headlines")

    st.markdown("""
    ---
    <div class="warning-box">
    <strong>Important context</strong><br>
    This model classifies headlines based on learned text patterns. It does <b>NOT</b> establish objective truth, fact-check claims, understand context, or replace professional fact-checking.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif page == "📊 Model Performance":
    page_header("EVALUATION", "Model performance", f"Actual results from the trained classifiers on a held-out test set of {evaluation['dataset']['test_samples']:,} headlines.")

    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        metric_card("Test accuracy", f"{selected_test['accuracy']:.2%}", "Held-out test set")
    with col2:
        metric_card("Fake-class F1", f"{selected_test['class_metrics']['fake']['f1']:.2%}", "Primary selection metric")
    with col3:
        metric_card("Balanced accuracy", f"{selected_test['balanced_accuracy']:.2%}", "Accounts for imbalance")
    with col4:
        metric_card("ROC-AUC", f"{selected_test['roc_auc']:.2%}", "Decision-score ranking")

    st.markdown('<div class="insight-strip"><strong>Key insight</strong><span>The model performs well on the majority Real class, while Fake-class detection remains the main challenge.</span></div>', unsafe_allow_html=True)

    st.subheader("Model comparison")
    df_comparison = pd.DataFrame([
        {
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Macro F1': (result['f1_real'] + result['f1_fake']) / 2,
            'Fake F1': result['f1_fake'],
            'Balanced Accuracy': result['balanced_accuracy'],
            'ROC-AUC': result['roc_auc']
        }
        for result in evaluation['models']
    ])
    comparison_long = df_comparison.melt(
        id_vars='Model',
        value_vars=['Accuracy', 'Macro F1', 'Fake F1', 'Balanced Accuracy'],
        var_name='Metric',
        value_name='Score'
    )
    comparison_chart = (
        alt.Chart(comparison_long)
        .mark_bar(size=18)
        .encode(
            x=alt.X('Model:N', title='Model', axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset('Metric:N', sort=['Accuracy', 'Macro F1', 'Fake F1', 'Balanced Accuracy']),
            y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                'Metric:N',
                title='Metric',
                scale=alt.Scale(range=['#63c7b2', '#91a9c4', '#d9b878', '#b48d9a'])
            ),
            tooltip=[alt.Tooltip('Model:N'), alt.Tooltip('Metric:N'), alt.Tooltip('Score:Q', format='.2%')]
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(comparison_chart, width="stretch")
    st.caption("Grouped comparison of the four primary evaluation metrics. Higher is better.")
    percentage_columns = ['Accuracy', 'Macro F1', 'Fake F1', 'Balanced Accuracy', 'ROC-AUC']
    st.dataframe(df_comparison.style.format({column: '{:.2%}' for column in percentage_columns}), width="stretch", hide_index=True)

    st.subheader("Class performance")
    class_metrics = selected_test['class_metrics']
    df_class = pd.DataFrame({
        'Class': ['Fake', 'Real'],
        'Precision': [class_metrics['fake']['precision'], class_metrics['real']['precision']],
        'Recall': [class_metrics['fake']['recall'], class_metrics['real']['recall']],
        'F1-score': [class_metrics['fake']['f1'], class_metrics['real']['f1']]
    }).set_index('Class')
    class_long = df_class.reset_index().melt(id_vars='Class', var_name='Metric', value_name='Score')
    class_chart = (
        alt.Chart(class_long)
        .mark_bar(size=34)
        .encode(
            x=alt.X('Class:N', title='Class', axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset('Metric:N', sort=['Precision', 'Recall', 'F1-score']),
            y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                'Metric:N',
                title='Metric',
                scale=alt.Scale(range=['#63c7b2', '#91a9c4', '#d9b878'])
            ),
            tooltip=[alt.Tooltip('Class:N'), alt.Tooltip('Metric:N'), alt.Tooltip('Score:Q', format='.2%')]
        )
        .properties(height=235)
        .interactive()
    )
    st.altair_chart(class_chart, width="stretch")
    st.caption("The Real class is the majority class; Fake-class scores are the primary risk signal.")

    st.subheader("Confusion matrix")
    cm = np.array(selected_test['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5.8, 3.25))
    fig.patch.set_facecolor('#101719')
    ax.set_facecolor('#172124')
    sns.heatmap(cm, annot=True, fmt=',d', cmap=sns.dark_palette('#63c7b2', as_cmap=True), cbar=False, linewidths=1, linecolor='#2b3b3e', ax=ax, xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], annot_kws={'color': '#e7ecef', 'fontsize': 13, 'weight': 'bold'})
    ax.set_xlabel('Predicted label', color='#9aa8ae')
    ax.set_ylabel('Actual label', color='#9aa8ae')
    ax.tick_params(colors='#c5ced1')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(f"{cm[0][1]:,} Fake headlines were classified as Real, making missed Fake headlines the main error type.")

    baseline = evaluation['baseline']
    st.subheader("Baseline comparison")
    baseline_col1, baseline_col2, baseline_col3 = st.columns(3, gap="small")
    with baseline_col1:
        metric_card("Majority baseline", f"{baseline['accuracy']:.2%}", "Always predict Real")
    with baseline_col2:
        metric_card("Final model", f"{selected_test['accuracy']:.2%}", evaluation['selection']['selected_model'])
    with baseline_col3:
        metric_card("Improvement", f"+{selected_test['accuracy'] - baseline['accuracy']:.2%}", "Percentage points")

    st.subheader("5-fold stratified cross-validation")
    cv_data = {
        'Metric': ['Accuracy', 'Macro F1', 'Fake F1', 'Balanced Accuracy'],
        'Mean ± standard deviation': [
            f"{selected_cv['accuracy_mean']:.2%} ± {selected_cv['accuracy_std']:.2%}",
            f"{selected_cv['f1_macro_mean']:.2%} ± {selected_cv['f1_macro_std']:.2%}",
            f"{selected_cv['f1_fake_mean']:.2%} ± {selected_cv['f1_fake_std']:.2%}",
            f"{selected_cv['balanced_accuracy_mean']:.2%} ± {selected_cv['balanced_accuracy_std']:.2%}"
        ]
    }
    st.dataframe(pd.DataFrame(cv_data), width="stretch", hide_index=True)
    st.caption("Mean ± standard deviation summarizes consistency across the five training folds; the held-out test set is not used here.")

# ============================================================================
# PAGE 3: FEATURE IMPORTANCE
# ============================================================================
elif page == "🔍 Feature Importance":
    page_header("INTERPRETABILITY", "Feature importance", "A transparent view of the language signals learned by the selected Linear SVC model.")
    
    st.markdown(f"""
    The selected model ({evaluation['selection']['selected_model']}) is a linear classifier trained on TF-IDF features.
    The coefficients reflect which words/phrases the model learned to associate with
    each class.
    
    **Important:** These features are model-level patterns learned from the training data.
    They do NOT prove that these words make news fake or real in reality.
    """)
    
    st.info("""
    **What This Shows:**
    - Words/phrases with high positive coefficients → associated with Real news in training data
    - Words/phrases with high negative coefficients → associated with Fake news in training data
    
    **What This Does NOT Show:**
    - Whether these words actually cause news to be fake or real
    - Proof of misinformation detection ability beyond the dataset
    - Causal relationships (correlation ≠ causation)
    """)
    
    col1, col2 = st.columns(2)
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    real_indices = np.argsort(coefficients)[-10:][::-1]
    fake_indices = np.argsort(coefficients)[:10]
    real_features = pd.DataFrame({
        'Feature': feature_names[real_indices],
        'Coefficient': coefficients[real_indices]
    })
    fake_features = pd.DataFrame({
        'Feature': feature_names[fake_indices],
        'Coefficient': coefficients[fake_indices]
    })

    with col1:
        st.subheader("✓ Real News Indicators")
        st.dataframe(real_features.style.format({'Coefficient': '{:.4f}'}), width="stretch", hide_index=True)
    
    with col2:
        st.subheader("⚠️ Fake News Indicators")
        st.dataframe(fake_features.style.format({'Coefficient': '{:.4f}'}), width="stretch", hide_index=True)
    
    st.warning("""
    **Key Insight:** The model learned that fake news in this dataset heavily features
    celebrity/entertainment gossip (Pattinson, Caitlyn, etc.). This pattern may not
    generalize to other types of fake news or datasets.
    """)
    
    st.markdown("""
    ---
    ### How Model Explanability Works
    
    For a linear model like Linear SVC with TF-IDF features:
    1. Each feature (word) has a coefficient (weight)
    2. Positive weights → push prediction toward "Real"
    3. Negative weights → push prediction toward "Fake"
    4. Larger absolute values → more important to the prediction
    
    This is useful for understanding model behavior but should not be interpreted as
    proof of feature causality or truth about the world.
    """)

# ============================================================================
# PAGE 4: BATCH PREDICTION
# ============================================================================
elif page == "📥 Batch Prediction":
    page_header("WORKFLOW", "Batch prediction", "Score multiple headlines from a CSV, inspect the results, and export them for review.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV must contain a 'title' column"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate
            if 'title' not in df.columns:
                st.error(f"❌ CSV must contain a 'title' column. Found columns: {list(df.columns)}")
            elif len(df) == 0:
                st.error("❌ CSV is empty.")
            else:
                st.success(f"✓ Loaded {len(df)} rows")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), width="stretch")
                
                # Run predictions
                if st.button("Run predictions on all rows", width="stretch"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = []
                    for idx, row in df.iterrows():
                        title = row['title']
                        if pd.isna(title) or not str(title).strip():
                            predictions.append({
                                'title': title,
                                'prediction': 'ERROR',
                                'prediction_label': 'Error',
                                'confidence': None
                            })
                            progress = (idx + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processed {idx + 1} / {len(df)}")
                            continue
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
                                'prediction': str(pred),
                                'prediction_label': pred_label,
                                'confidence': conf
                            })
                        
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {idx + 1} / {len(df)}")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    df_results = pd.DataFrame(predictions)
                    
                    st.subheader("Prediction Results")
                    st.dataframe(df_results, width="stretch", hide_index=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    
                    fake_count = (df_results['prediction_label'] == 'Fake').sum()
                    real_count = (df_results['prediction_label'] == 'Real').sum()
                    error_count = (df_results['prediction_label'] == 'Error').sum()
                    
                    with col1:
                        st.metric("Predicted Fake", fake_count)
                    with col2:
                        st.metric("Predicted Real", real_count)
                    with col3:
                        st.metric("Errors", error_count)
                    
                    # Download results
                    csv_buffer = io.StringIO()
                    df_results.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="predictions_results.csv",
                        mime="text/csv",
                        width="stretch"
                    )
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("Upload a CSV file with a `title` column to begin.")
        
        # Example format
        st.subheader("Example CSV Format")
        example_df = pd.DataFrame({
            'title': [
                'Scientists discover new renewable energy source',
                'Celebrity breaks up with mysterious stranger',
                'New study shows health benefits of exercise',
                'Breaking: Major discovery announced'
            ]
        })
        st.dataframe(example_df, width="stretch", hide_index=True)
        
        # Download template
        csv_template = example_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Example CSV Template",
            data=csv_template,
            file_name="example_input.csv",
            mime="text/csv",
            width="stretch"
        )

# ============================================================================
# PAGE 5: PROJECT INFORMATION
# ============================================================================
elif page == "ℹ️ Project Information":
    page_header("PROJECT NOTES", "About Fake News Detector", "A portfolio project that demonstrates an end-to-end text classification workflow, with its boundaries made explicit.")
    
    st.subheader("Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{evaluation['dataset']['total_samples']:,}")
    with col2:
        real_count = evaluation['dataset']['class_counts']['real']
        st.metric("Real News", f"{real_count:,} ({real_count / evaluation['dataset']['total_samples']:.1%})")
    with col3:
        fake_count = evaluation['dataset']['class_counts']['fake']
        st.metric("Fake News", f"{fake_count:,} ({fake_count / evaluation['dataset']['total_samples']:.1%})")
    with col4:
        st.metric("Class Imbalance", f"{real_count / fake_count:.2f}:1")
    
    st.markdown(f"""
    **Dataset Name:** {evaluation['dataset']['name']}
    
    **Source:** News headlines with labels (Real or Fake)
    
    **Features:** Title text only
    
    **Split:** 80% train ({evaluation['dataset']['train_samples']:,}) / 20% test ({evaluation['dataset']['test_samples']:,})
    """)
    
    st.subheader("NLP Approach")
    st.markdown(f"""
    **Text Preprocessing:**
    1. Convert to lowercase
    2. Remove URLs
    3. Remove special characters
    4. Normalize whitespace
    
    **Feature Extraction:** TF-IDF Vectorization
    - N-grams: {vectorizer.ngram_range[0]}-{vectorizer.ngram_range[1]}
    - Max features: {vectorizer.max_features:,}
    - Min document frequency: {vectorizer.min_df}
    - Max document frequency: {vectorizer.max_df:.0%}
    - Stop words: {vectorizer.stop_words}
    
    **Classifier:** {evaluation['selection']['selected_model']} (Support Vector Machine)
    - Kernel: Linear
    - Optimization: Dual formulation off (faster)
    - Random state: {model.random_state} (reproducible)
    """)
    
    st.subheader("Models Evaluated")
    st.dataframe(
        pd.DataFrame([
            {'Model': result['model_name'], 'Test Fake F1': result['f1_fake']}
            for result in evaluation['models']
        ]),
        width="stretch",
        hide_index=True
    )
    st.caption(f"Selection criterion: {evaluation['selection']['criterion']}")
    
    st.subheader("Key Limitations")
    
    st.markdown("""
    <div class="limitation-box">
    <strong>1. Title-Only Features</strong><br>
    The model uses only headline text, ignoring article body, source credibility,
    publication metadata, and other contextual information crucial for real fact-checking.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="limitation-box">
    <strong>2. Pattern Matching, Not Truth Detection</strong><br>
    The model learns statistical patterns in the training data, not objective truth.
    Different datasets would produce different patterns.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="limitation-box">
    <strong>3. Class Imbalance</strong><br>
    Fake news is underrepresented ({fake_count / evaluation['dataset']['total_samples']:.1%} vs {real_count / evaluation['dataset']['total_samples']:.1%} real). The model is biased
    toward predicting "Real", reducing Fake-class recall to {selected_test['class_metrics']['fake']['recall']:.1%}.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="limitation-box">
    <strong>4. Fake-class false negatives</strong><br>
    {selected_test['confusion_matrix'][0][1] / (selected_test['confusion_matrix'][0][0] + selected_test['confusion_matrix'][0][1]):.1%} of Fake headlines are misclassified as Real ({selected_test['confusion_matrix'][0][1]} out of {selected_test['confusion_matrix'][0][0] + selected_test['confusion_matrix'][0][1]} test-set Fake headlines).
    These are false negatives for the Fake class, reducing Fake-class recall.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="limitation-box">
    <strong>5. Temporal Validity</strong><br>
    Fake news tactics evolve. The model is based on historical data and may
    become outdated without periodic retraining.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Use Cases & Limitations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Appropriate Uses")
        st.markdown("""
        - Research and analysis
        - Educational demonstrations
        - Assisting human reviewers
        - A/B testing with feedback
        - Portfolio projects
        """)
    
    with col2:
        st.markdown("#### ❌ NOT Appropriate For")
        st.markdown("""
        - Autonomous content removal
        - Fact-checking without review
        - Regulatory decisions
        - Mission-critical applications
        - Systems without human oversight
        """)
    
    st.markdown("""
    ---
    <div class="warning-box">
    <strong>⚠️ CRITICAL DISCLAIMER</strong>
    <br>
    This model is a machine learning classifier that learned patterns from training data.
    It should NEVER be used as a sole authority on whether content is true or false.
    
    News that is "factually accurate" may still be misleading through selective reporting.
    News that is "untrue" may be spread inadvertently by genuine journalists.
    
    Proper fact-checking requires:
    - Domain expertise
    - Multiple reliable sources
    - Cross-reference verification
    - Contextual understanding
    - Human judgment
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Technical Stack")
    st.markdown("""
    - **Framework:** Streamlit
    - **ML Library:** scikit-learn
    - **Data Processing:** pandas, numpy
    - **Language:** Python 3.9+
    """)
    
    st.subheader("Project Files")
    st.markdown("""
    - `app.py` - Streamlit application
    - `final_model.pkl` - Trained Linear SVC model
    - `final_vectorizer.pkl` - Fitted TF-IDF vectorizer
    - `execute_pipeline.py` - ML pipeline execution script
    - `requirements.txt` - Python dependencies
    - `README.md` - Project documentation
    """)
