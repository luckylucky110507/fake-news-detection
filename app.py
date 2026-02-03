import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main {
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #1f77b4;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-real {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .result-fake {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_models():
    """Load the trained model, vectorizer, and encoder."""
    try:
        model_path = SCRIPT_DIR / "fake_news_model.pkl"
        vectorizer_path = SCRIPT_DIR / "tfidf_vectorizer.pkl"
        encoder_path = SCRIPT_DIR / "label_encoder.pkl"

        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            st.info("Required file: fake_news_model.pkl")
            st.stop()
        if not vectorizer_path.exists():
            st.error(f"Vectorizer file not found at: {vectorizer_path}")
            st.info("Required file: tfidf_vectorizer.pkl")
            st.stop()
        if not encoder_path.exists():
            st.error(f"Encoder file not found at: {encoder_path}")
            st.info("Required file: label_encoder.pkl")
            st.stop()

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        return model, vectorizer, encoder
    except pickle.UnpicklingError as exc:
        st.error(f"Error unpickling model: {exc}")
        st.info("The model files may be corrupted. Please retrain the model.")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error loading models: {exc}")
        st.stop()


try:
    model, vectorizer, encoder = load_models()
except Exception as exc:
    st.error(f"Failed to load models: {exc}")
    st.stop()


def clean_text(text: str) -> str:
    """Clean text: same processing used during training."""
    try:
        text = str(text).lower().strip()
        if not text:
            return ""
        text = "".join(c if c.isalpha() or c.isspace() else " " for c in text)
        text = " ".join(text.split())
        return text
    except Exception as exc:
        st.error(f"Error cleaning text: {exc}")
        return ""


def predict_news(text: str):
    """Predict if news is fake or real."""
    try:
        cleaned = clean_text(text)

        if not cleaned:
            return None, None, "The text contains no valid words after cleaning."

        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]
        label = encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probability) * 100)

        return label, confidence, None
    except Exception as exc:
        return None, None, f"Prediction error: {exc}"


# Main UI
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<h1 style='text-align: center; color: #dc3545;'>FAKE NEWS DETECTION</h1>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About This System")
    st.write(
        """
    This system detects whether a news article is FAKE or REAL
    using a trained machine learning model.

    How it works:
    - Analyzes the text content
    - Compares patterns with trained data
    - Returns a classification result
    """
    )

    st.header("Model Info")
    try:
        st.info(
            f"""
        Model Status: Loaded Successfully
        Classes: {', '.join(map(str, encoder.classes_))}
        Accuracy: 96.81%
        """
        )
    except Exception as exc:
        st.error(f"Error displaying model info: {exc}")

    st.header("Tips")
    st.write(
        """
    - Paste the full article text for better accuracy
    - Longer articles provide more context
    - The model is trained on thousands of articles
    """
    )

# Main content
st.markdown("### Paste News Article Below")

try:
    news_text = st.text_area(
        "Enter news text to check:",
        height=200,
        placeholder="Paste or type the news article here...",
        label_visibility="collapsed",
    )
except Exception as exc:
    st.error(f"Error with text area: {exc}")
    news_text = ""

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    check_button = st.button(
        "Analyze News", use_container_width=True, type="primary"
    )

if check_button:
    if not news_text or not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            with st.spinner("Analyzing..."):
                label, confidence, error = predict_news(news_text)

                if error:
                    st.error(f"Error: {error}")
                elif label is None or confidence is None:
                    st.error("Could not make prediction. Please try again.")
                else:
                    st.markdown("---")

                    if str(label).lower() == "real":
                        st.markdown(
                            f"""
                        <div class='result-box result-real'>
                        <h2 style='color: #28a745;'>REAL NEWS</h2>
                        <p style='font-size: 18px;'>Confidence: <strong>{confidence:.2f}%</strong></p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif str(label).lower() == "fake":
                        st.markdown(
                            f"""
                        <div class='result-box result-fake'>
                        <h2 style='color: #dc3545;'>FAKE NEWS</h2>
                        <p style='font-size: 18px;'>Confidence: <strong>{confidence:.2f}%</strong></p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(f"Unknown prediction: {label}")

                    st.markdown("---")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", str(label).upper())
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")
        except Exception as exc:
            st.error(f"Unexpected error during analysis: {exc}")

st.markdown("---")

# Footer
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Important Disclaimer:</strong></p>
    <p>This tool is designed to assist in news verification but should not be relied upon as the sole source of truth.
    Always cross-reference multiple sources for critical information.</p>
    <hr>
    <small>Fake News Detection System | Powered by Machine Learning</small>
</div>
""",
    unsafe_allow_html=True,
)
