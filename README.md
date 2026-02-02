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
fake news detection/
├── app.py                      # Streamlit web interface
├── predict.py                  # CLI prediction tool
├── train_model.py              # Original training script
├── train_fast.py              # Optimized fast training script
├── test_model.py              # Model testing script
├── fake_real_news.csv         # Dataset (44,898 articles)
├── requirements.txt           # Python dependencies
├── fake_news_model.pkl        # Trained model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── label_encoder.pkl          # Label encoder
└── README.md                  # This file
```

---

## 🚀 Quick Start

### 1. Setup Python Environment

**Option A: Using Virtual Environment (Recommended)**

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**Option B: Using Conda**

```bash
conda create -n fake-news python=3.10
conda activate fake-news
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (First Time Only)

Choose the training script based on your needs:

**Fast Training (Recommended - ~2 minutes, 96.81% accuracy):**
```bash
python train_fast.py
```

**Full Training (Takes longer, trains on entire dataset):**
```bash
python train_model.py
```

Output:
- `fake_news_model.pkl` - Trained ML model
- `tfidf_vectorizer.pkl` - Text vectorizer
- `label_encoder.pkl` - Label encoder

### 4. Use the Model

#### Option A: Web Interface (Streamlit) ⭐

```bash
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

**Features:**
- ✅ Clean, user-friendly interface
- ✅ Real-time predictions
- ✅ Confidence scores
- ✅ Beautiful visualizations

#### Option B: Command Line Interface

```bash
python predict.py
```

**Interactive menu:**
```
1. Predict news
2. Exit
```

Paste your news text and get instant predictions!

#### Option C: Python Script

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

## 🧪 Testing

### Run Tests

```bash
python test_model.py
```

**Test Samples:**
```
✓ News about elections → Predicted correctly
✓ Conspiracy theories → Predicted as fake
✓ Medical studies → Predicted correctly
✓ Vaccine rumors → Predicted as fake
```

---

## 🔧 Troubleshooting

### Models not loading
```
Error: Model file not found
Solution: Run train_fast.py or train_model.py to generate models
```

### Streamlit not starting
```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit --upgrade
```

### Slow predictions
```
Solution: This is normal for large datasets. Train on sampled data:
python train_fast.py
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
| `app.py` | Streamlit web interface for predictions |
| `predict.py` | Interactive CLI for predictions |
| `train_model.py` | Complete training script |
| `train_fast.py` | Fast training on sampled data |
| `test_model.py` | Test predictions with samples |
| `fake_real_news.csv` | Dataset with 44,898 articles |
| `requirements.txt` | All Python dependencies |

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
