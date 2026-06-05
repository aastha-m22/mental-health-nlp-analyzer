"""
app/streamlit_app.py
--------------------
Streamlit web application for real-time mental health text analysis.

Run: streamlit run app/streamlit_app.py

Features:
  - Text input with real-time prediction
  - Confidence score bars
  - Top contributing words (TF-IDF weight proxy)
  - Sentiment analysis display
  - Ethical disclaimer
"""

import streamlit as st
import numpy as np
import joblib
import os
import sys
import re

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import get_preprocessor
from src.features import TFIDFExtractor, extract_sentiment_features, extract_linguistic_features
from src.nlp_enhancements import batch_linguistic_patterns, batch_emotion_scores
from textblob import TextBlob

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health NLP Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

LABEL_NAMES = ["Normal", "Depression", "Anxiety"]
LABEL_EMOJIS = {"Normal": "✅", "Depression": "💙", "Anxiety": "🌊"}
LABEL_COLORS = {"Normal": "#2ecc71", "Depression": "#3498db", "Anxiety": "#e74c3c"}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEAT_DIR = os.path.join(MODEL_DIR, "feature_extractors")

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .prediction-box {
        padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
        border-left: 6px solid; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border: 1px solid #e0e0e0; text-align: center;
    }
    .disclaimer {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 8px; padding: 1rem; font-size: 0.85rem;
    }
    .confidence-bar {
        height: 22px; border-radius: 4px; margin: 4px 0;
        display: flex; align-items: center; padding-left: 8px;
        color: white; font-weight: 600; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model & feature extractors ──────────────────────────────────
@st.cache_resource
def load_model_and_extractors():
    """Load the saved model and TF-IDF vectorizer."""
    try:
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        tfidf_path = os.path.join(FEAT_DIR, "tfidf.pkl")

        if not os.path.exists(model_path):
            return None, None, None, None, "Model not found. Run train_pipeline.py first."
        if not os.path.exists(tfidf_path):
            return None, None, None, None, "TF-IDF vectorizer not found. Run train_pipeline.py first."

        model = joblib.load(model_path)
        tfidf_vec = joblib.load(tfidf_path)

        # Load model name
        name_path = os.path.join(MODEL_DIR, "best_model_name.txt")
        model_name = open(name_path).read().strip() if os.path.exists(name_path) else "Unknown"

        labels_path = os.path.join(MODEL_DIR, "label_names.txt")
        label_names = LABEL_NAMES
        if os.path.exists(labels_path):
            label_names = [
                line.strip().replace("_", " ").title()
                for line in open(labels_path).read().splitlines()
                if line.strip()
            ]

        return model, tfidf_vec, model_name, label_names, None
    except Exception as e:
        return None, None, None, None, str(e)


@st.cache_resource
def load_preprocessor():
    return get_preprocessor(method="lemmatize")


def build_features(text: str, tfidf_vec, preprocessor):
    """Build the same feature vector used during training."""
    cleaned = preprocessor.clean(text)
    tokens = preprocessor.tokenize(text)

    # TF-IDF
    tfidf_feats = tfidf_vec.transform([cleaned]).toarray()

    # Sentiment
    sent = extract_sentiment_features([cleaned])

    # Linguistic
    ling = extract_linguistic_features([cleaned])

    # NLP patterns
    patterns = batch_linguistic_patterns([cleaned])
    emotions = batch_emotion_scores([tokens])

    X = np.hstack([tfidf_feats, sent, ling, patterns, emotions])
    return X, cleaned, tokens


def get_top_tfidf_words(text: str, tfidf_vec, n: int = 8):
    """Return top N words by TF-IDF score for the input text."""
    cleaned = text.lower()
    tfidf_matrix = tfidf_vec.transform([cleaned])
    feature_names = tfidf_vec.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    top_idx = np.argsort(scores)[-n:][::-1]
    return [(feature_names[i], scores[i]) for i in top_idx if scores[i] > 0]


def render_confidence_bars(probabilities, label_names=None):
    """Render HTML confidence bars."""
    label_names = label_names or LABEL_NAMES
    bars_html = ""
    fallback_colors = ["#2ecc71", "#3498db", "#e74c3c", "#9467bd", "#f58518"]
    for i, (label, prob) in enumerate(zip(label_names, probabilities)):
        color = LABEL_COLORS.get(label, fallback_colors[i % len(fallback_colors)])
        width = int(prob * 100)
        bars_html += f"""
        <div style="margin: 6px 0;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:90px; font-size:0.85rem; font-weight:600;">{label}</span>
                <div style="flex:1; background:#f0f0f0; border-radius:6px; height:22px;">
                    <div style="width:{width}%; background:{color}; border-radius:6px;
                                height:22px; display:flex; align-items:center;
                                padding-left:8px; color:white; font-size:0.8rem; font-weight:700;
                                min-width: 30px;">
                        {prob*100:.1f}%
                    </div>
                </div>
            </div>
        </div>
        """
    return bars_html


# ─────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=70)
        st.markdown("### 🧠 Mental Health NLP")
        st.markdown("---")
        st.markdown("""
        **About this tool**

        Uses NLP + Machine Learning to detect potential signs of:
        - 💙 **Depression** — hopelessness, low energy, withdrawal
        - 🌊 **Anxiety** — worry, panic, hypervigilance
        - ✅ **Normal** — neutral/positive expression

        **Tech Stack:**
        - Python, scikit-learn, NLTK
        - TF-IDF, Word2Vec, Sentiment
        - Streamlit deployment
        """)
        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <strong>Disclaimer</strong><br>
        This is an educational ML project.
        <strong>Not a clinical tool.</strong>
        If you or someone you know is struggling,
        please contact a mental health professional
        or call a crisis helpline.
        </div>
        """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────
    st.markdown('<p class="main-header">🧠 Mental Health Text Analyzer</p>', unsafe_allow_html=True)
    st.markdown("Detect potential signs of **Depression** or **Anxiety** from social media text using NLP.")
    st.markdown("---")

    # ── Load model ─────────────────────────────────────────────────
    model, tfidf_vec, model_name, label_names, error = load_model_and_extractors()
    preprocessor = load_preprocessor()

    if error:
        st.error(f"⚠️ {error}")
        st.info("**Quick Start:** Run `python notebooks/train_pipeline.py` to train and save the model first.")
        st.stop()

    st.success(f"✅ Model loaded: **{model_name}**")

    # ── Input ──────────────────────────────────────────────────────
    st.markdown("### ✍️ Enter Text to Analyze")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_text = st.text_area(
            "Paste a social media post, tweet, or any text:",
            height=130,
            placeholder="e.g., 'I've been feeling so empty lately, can't seem to find joy in anything I used to love...'",
            key="input_text",
        )

    with col2:
        st.markdown("**Try examples:**")
        if st.button("😔 Depression"):
            st.session_state.input_text = "I can't seem to get out of bed anymore. Everything feels heavy and pointless. I've lost interest in things I used to love."
            st.rerun()
        if st.button("😰 Anxiety"):
            st.session_state.input_text = "My heart is racing and I can't stop thinking about worst-case scenarios. I had to leave the meeting because I couldn't breathe properly."
            st.rerun()
        if st.button("😊 Normal"):
            st.session_state.input_text = "Had a wonderful day hiking with friends! The weather was perfect and we found an amazing waterfall."
            st.rerun()

    analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)

    if analyze_btn and user_text.strip():
        with st.spinner("Analyzing..."):
            try:
                # Build features
                X, cleaned_text, tokens = build_features(user_text, tfidf_vec, preprocessor)

                # Predict
                prediction = model.predict(X)[0]
                label = label_names[prediction] if label_names and prediction < len(label_names) else LABEL_NAMES[prediction]
                emoji = LABEL_EMOJIS.get(label, "")
                color = LABEL_COLORS.get(label, "#4c78a8")

                # Probabilities
                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)[0]
                elif hasattr(model, "decision_function"):
                    df = model.decision_function(X)[0]
                    # Softmax approximation
                    exp_df = np.exp(df - np.max(df))
                    probabilities = exp_df / exp_df.sum()

                # Top TF-IDF words
                top_words = get_top_tfidf_words(cleaned_text, tfidf_vec)

                # Sentiment
                blob = TextBlob(user_text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # ── Results ─────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")

                # Main prediction card
                st.markdown(f"""
                <div class="prediction-box" style="border-color:{color}; background:{color}15;">
                    <h2 style="margin:0; color:{color};">{emoji} {label}</h2>
                    <p style="margin:4px 0 0; color:#555; font-size:0.95rem;">
                        Primary classification based on NLP pattern analysis
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    sentiment_label = "Positive 😊" if polarity > 0.1 else ("Negative 😔" if polarity < -0.1 else "Neutral 😐")
                    st.metric("Sentiment Polarity", f"{polarity:.3f}", sentiment_label)
                with col_b:
                    st.metric("Subjectivity", f"{subjectivity:.3f}", "Personal expression" if subjectivity > 0.5 else "Factual tone")
                with col_c:
                    word_count = len(user_text.split())
                    st.metric("Word Count", word_count)

                # Confidence bars
                if probabilities is not None:
                    st.markdown("#### 📈 Confidence Scores")
                    st.markdown(render_confidence_bars(probabilities, label_names), unsafe_allow_html=True)

                # Top words
                if top_words:
                    st.markdown("#### 🔑 Key Contributing Words")
                    word_cols = st.columns(min(len(top_words), 4))
                    for i, (word, score) in enumerate(top_words[:4]):
                        with word_cols[i % 4]:
                            st.markdown(f"""
                            <div class="metric-card">
                               <div style="font-size:1.1rem; font-weight:700;">{word}</div>
                               <div style="font-size:0.8rem; color:#888;">score: {score:.4f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Preprocessed text
                with st.expander("🔧 Preprocessed Text"):
                    st.code(cleaned_text, language=None)
                    st.caption("After cleaning, stopword removal, and lemmatization")

                # Ethical note
                st.markdown("---")
                st.markdown("""
                <div class="disclaimer">
                ⚕️ <strong>Important:</strong> This tool is for educational and research purposes only.
                Mental health diagnosis requires a qualified professional.
                If you're concerned about someone's wellbeing, encourage them to seek help from
                a licensed therapist or call a crisis helpline.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")

    # ── Model Info ─────────────────────────────────────────────────
    with st.expander("ℹ️ About the Model & Methodology"):
        st.markdown(f"""
        **Model:** {model_name}

        **Feature Pipeline:**
        - TF-IDF (5,000 features, unigrams + bigrams, sublinear scaling)
        - Sentiment polarity & subjectivity (TextBlob)
        - Linguistic features (word count, punctuation density, etc.)
        - NLP patterns (absolutist language, hedging, first-person ratio)
        - Emotion lexicon scores (sadness, fear, joy, anger, disgust)

        **Training Data:** Reddit-style mental health posts (simulated dataset)
        In production: SMHD, CLPsych, or similar validated datasets

        **Ethical Considerations:**
        - Model may be biased toward the language patterns in training data
        - Cultural and linguistic diversity is not fully represented
        - Should never replace clinical assessment
        - Privacy: no user data is stored or transmitted
        """)


if __name__ == "__main__":
    main()
