import json
import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def parse_news_file(path: str) -> pd.DataFrame:
    """
    The provided News_clean.csv file is not a normal CSV table – it is a
    semi-structured export where every article is spread over several lines like:

        {,,
        ,\"\"\"link\"\": \"...\",,
        ,\"\"\"headline\"\": \"...\",,
        ,\"\"\"category\"\": \"...\",,
        ...
        },,

    This helper parses that structure into a proper DataFrame with columns:
    link, headline, category, short_description, authors, date.
    """
    raw = pd.read_csv(path, header=None, dtype=str).fillna("")

    articles = []
    current_lines: list[str] = []

    for _, row in raw.iterrows():
        c0 = str(row[0]).strip()
        c1 = str(row[1])
        c2 = str(row[2])

        if c0 == "{":
            # Start of a new JSON object
            current_lines = ["{"]
            continue

        if c0 == "}":
            # End of current JSON object
            current_lines.append("}")
            block = "\n".join(current_lines)
            try:
                obj = json.loads(block)
                articles.append(obj)
            except Exception:
                # Ignore malformed blocks
                pass
            current_lines = []
            continue

        # Middle of an object: join the text fragments from columns 1 and 2
        line = (c1 + c2).strip()
        if not line:
            continue
        current_lines.append(line)

    df = pd.DataFrame(articles)
    return df


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    # 1. Parse the custom-formatted file into a clean table
    df = parse_news_file("News_clean.csv")
    print("Parsed rows:", len(df))
    print("Columns:", df.columns.tolist())

    if df.empty:
        raise ValueError("No data could be parsed from News_clean.csv")

    # 2. Build text + label columns
    df["headline"] = df["headline"].fillna("")
    df["short_description"] = df.get("short_description", "").fillna("")
    df["category"] = df["category"].fillna("")

    df["text"] = (df["headline"] + " " + df["short_description"]).str.strip()

    # Drop rows without text or label
    df = df[(df["text"] != "") & (df["category"] != "")]
    print("Usable rows after cleaning:", len(df))

    # 3. Clean text
    df["text"] = df["text"].apply(clean_text)

    # 4. Encode labels (news category)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["category"])

    # 5. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2)
    X = vectorizer.fit_transform(df["text"])

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Train models
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # 8. Accuracy comparison
    accuracies = {
        "LogisticRegression": accuracy_score(y_test, y_pred_lr),
        "NaiveBayes": accuracy_score(y_test, y_pred_nb),
        "RandomForest": accuracy_score(y_test, y_pred_rf),
    }

    for model, acc in accuracies.items():
        print(f"{model} Accuracy: {acc:.4f}")

    best_model_name = max(accuracies, key=accuracies.get)
    print(f"\nBest Model: {best_model_name}")

    best_model = {
        "LogisticRegression": lr,
        "NaiveBayes": nb,
        "RandomForest": rf,
    }[best_model_name]

    # 9. Save model, vectorizer, and encoder
    with open("fake_news_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("\nModel, Vectorizer, and Encoder saved successfully.")


if __name__ == "__main__":
    main()

