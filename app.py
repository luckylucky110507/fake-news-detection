import streamlit as st
import pickle
import re

# Load model, vectorizer and label encoder
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))


def clean_text(text: str) -> str:
    """Same cleaning used during training."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


st.title("📰 News Category Classification")

news = st.text_area("Enter News Text")

if st.button("Classify"):
    if news.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(news)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)
        label = encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted category: {label}")
