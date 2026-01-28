import pickle
import re

# Load model, vectorizer and label encoder
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


def clean_text(text: str) -> str:
    """Same cleaning used during training."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    news = input("Enter news text: ")

    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    result = encoder.inverse_transform(prediction)

    print("\nPredicted category:", result[0])


if __name__ == "__main__":
    main()

