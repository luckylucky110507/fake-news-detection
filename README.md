# Fake News Detection System

A machine learning project that detects whether news articles are FAKE or REAL using NLP and Logistic Regression.

Current Model Performance:
- Accuracy: 96.81%
- Precision: 96.82%
- Recall: 96.81%
- F1-Score: 96.81%

---

## Project Structure

```
fake-news-detection/
├── app.py
├── fake_real_news.csv
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml
```

---

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

Then open: `http://localhost:8501`

Features:
- Real-time predictions
- Confidence scores
- Color-coded results
- Sidebar model info

---

## Dataset

Source: `fake_real_news.csv`

Statistics:
- Total articles: 44,898
- Fake news: 23,481 (52.3%)
- Real news: 21,417 (47.7%)
- Columns: `text`, `label`

---

## Model Details

Algorithm: Logistic Regression

Features:
- TF-IDF Vectorization
- Unigrams
- 2,000 features selected
- Max frequency: 95%

Text Cleaning:
1. Convert to lowercase
2. Remove special characters (keep letters and spaces)
3. Remove extra whitespace
4. Strip leading/trailing spaces

---

## Dependencies

See `requirements.txt` for pinned versions.

---

## Troubleshooting

Models not loading:
```
Error: Model file not found
Solution: Ensure all .pkl files are in the project folder
```

Streamlit not starting:
```bash
pip install streamlit --upgrade
```

Port 8501 already in use:
```bash
streamlit run app.py --server.port 8502
```

---

## Notes

- This tool assists in verification but should not be the sole source of truth.
- Cross-reference multiple sources for critical information.

