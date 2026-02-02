# 🚨 Fake News Detection System

A complete machine learning project that detects whether news articles are **FAKE** or **REAL** using Natural Language Processing and Logistic Regression.

**Current Model Performance:**
- ✅ **Accuracy: 96.81%**
- ✅ **Precision: 96.82%**
- ✅ **Recall: 96.81%**
- ✅ **F1-Score: 96.81%**

---

## 📋 Project Structure

```
fake-news-detection/
├── app.py                      # Streamlit web interface
├── fake_real_news.csv         # Dataset (44,898 articles)
├── fake_news_model.pkl        # Trained Logistic Regression model
├── tfidf_vectorizer.pkl       # TF-IDF text vectorizer
├── label_encoder.pkl          # Binary label encoder
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web App (Streamlit)

```bash
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

**Features:**
- ✅ Clean, user-friendly interface
- ✅ Real-time predictions
- ✅ Confidence scores
- ✅ Color-coded results (Green=Real, Red=Fake)
- ✅ Interactive sidebar information

### 3. Using the Model in Python

```python
import pickle
import numpy as np

# Load models
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Test news
news = "Your news article text here..."
cleaned = news.lower()
vector = vectorizer.transform([cleaned])
prediction = model.predict(vector)[0]
confidence = np.max(model.predict_proba(vector)[0]) * 100
label = encoder.inverse_transform([prediction])[0]

print(f"Prediction: {label} ({confidence:.2f}%)")
```

---

## 📊 Dataset

**Source:** `fake_real_news.csv`

**Statistics:**
- Total articles: **44,898**
- Fake news: **23,481** (52.3%)
- Real news: **21,417** (47.7%)
- Columns: `text`, `label`

---

## 🤖 Model Details

### Algorithm
**Logistic Regression** - Fast, interpretable, and highly accurate

### Features
- **TF-IDF Vectorization**
  - Stop words removed
  - Unigrams (single words)
  - 2,000 features selected
  - Min frequency: 1, Max frequency: 95%

### Training Process
1. Load and clean data
2. Sample 10,000 articles for efficiency
3. Vectorize text using TF-IDF
4. Split into train (80%) / test (20%)
5. Train Logistic Regression model
6. Evaluate using multiple metrics

### Text Cleaning
```python
1. Convert to lowercase
2. Remove special characters (keep letters & spaces)
3. Remove extra whitespace
4. Strip leading/trailing spaces
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 96.81% |
| Precision | 96.82% |
| Recall | 96.81% |
| F1-Score | 96.81% |

**Interpretation:**
- Model correctly classifies ~97 out of 100 articles
- Balanced performance for both fake and real news
- Highly reliable predictions

---

## 📚 File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit web interface for predictions |
| `fake_real_news.csv` | Dataset with 44,898 labeled articles |
| `fake_news_model.pkl` | Pre-trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (2000 features) |
| `label_encoder.pkl` | Binary label encoder (FAKE=0, REAL=1) |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Git configuration for version control |
| `.streamlit/config.toml` | Streamlit app configuration |

---

## 🔧 Troubleshooting

### Models not loading
```
Error: Model file not found
Solution: Ensure all .pkl files are in the project folder
```

### Streamlit not starting
```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit --upgrade
```

### Port 8501 already in use
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computing |
| scikit-learn | >=1.3.0 | ML algorithms |
| nltk | >=3.8 | NLP tools |
| streamlit | >=1.28.0 | Web interface |
| altair | >=4.2.0 | Visualization |

---

## 💡 How It Works

### Prediction Pipeline

```
User Input Text
    ↓
Text Cleaning (lowercase, remove special chars)
    ↓
TF-IDF Vectorization (convert to numbers)
    ↓
Logistic Regression Model
    ↓
Probability Scores
    ↓
Classification: FAKE or REAL
    ↓
Display Results with Confidence
```

---

## ⚠️ Limitations & Disclaimer

1. **Model Accuracy**: ~97% but not 100%
2. **Dataset Bias**: Trained on specific dataset
3. **Language**: Works best with English text
4. **Context**: Cannot verify sources or facts
5. **Use Responsibly**: 
   - Don't rely solely on this model
   - Cross-reference multiple sources
   - Combine with human judgment

---

## 🎓 Learning Insights

### Key Concepts Demonstrated
- ✅ Text preprocessing and NLP
- ✅ Feature extraction (TF-IDF)
- ✅ Machine learning model training
- ✅ Model evaluation and metrics
- ✅ Web app development (Streamlit)
- ✅ Pickle serialization
- ✅ CLI and GUI interfaces

---

## 📚 File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit web interface for predictions |
| `fake_real_news.csv` | Dataset with 44,898 labeled articles |
| `fake_news_model.pkl` | Pre-trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (2000 features) |
| `label_encoder.pkl` | Binary label encoder (FAKE=0, REAL=1) |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Git configuration for version control |
| `.streamlit/config.toml` | Streamlit app configuration |

---

## 🔐 Security Notes

- Models are safe pickle files
- No external API calls
- All processing local
- Dataset is public

---

## 📞 Support

**To retrain the model:**
```bash
python train_fast.py    # Quick (~2 min)
python train_model.py   # Full training (~10+ min)
```

**To test model performance:**
```bash
python test_model.py
```

**To use the web app:**
```bash
streamlit run app.py
```

---

## 📄 License

This project is for educational purposes. Feel free to modify and use as needed.

---

## ✨ Features

- ✅ High accuracy (96.81%)
- ✅ Fast predictions (<1 second)
- ✅ Beautiful web interface
- ✅ CLI support
- ✅ Detailed metrics
- ✅ Easy to extend
- ✅ Well documented

---

**Last Updated:** February 2, 2026  
**Model Version:** 1.0  
**Status:** ✅ Production Ready
