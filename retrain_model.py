import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    if not text:
        return ""
    text = "".join(c if c.isalpha() or c.isspace() else " " for c in text)
    return " ".join(text.split())


def main():
    data_path = Path("fake_real_news.csv")
    if not data_path.exists():
        raise FileNotFoundError("fake_real_news.csv not found")

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    # Sample for faster training if dataset is large
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)

    df["text"] = df["text"].fillna("").map(clean_text)
    df = df[df["text"].str.len() > 0]

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        max_df=0.95,
        min_df=1,
    )
    X = vectorizer.fit_transform(df["text"])

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X, y)

    with open("fake_news_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("Saved fake_news_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl")


if __name__ == "__main__":
    main()
