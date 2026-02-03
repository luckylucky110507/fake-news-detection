# Streamlit Cloud Deployment Guide

## Fixed Deployment Issues

The following items are aligned for Streamlit Cloud:

1. Simplified `requirements.txt`
   - Only essential packages are listed with pinned versions.

2. Minimal Streamlit configuration
   - `.streamlit/config.toml` includes only required settings.
   - `.streamlit/secrets.toml` exists (empty, no secrets required).

3. Project structure
```
fake-news-detection/
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
└── .streamlit/
    ├── config.toml
    └── secrets.toml
```

## Deployment Steps

1. Go to Streamlit Cloud and open your app.
2. Click Settings (gear icon).
3. Click Reboot app or wait for auto-redeploy.
4. Wait for the build to finish.

## Expected Result

The app should build successfully and show the prediction UI.
