# News Classification (Fake News Detection Project)

This project trains a text classifier on a news dataset and exposes it through:

- a **Streamlit web app** (`app.py`)
- a **CLI prediction script** (`predict.py`)
- a **training script** (`train_model.py`)

> Note: With the provided `News_clean.csv` file, the model predicts **news categories** (e.g. `U.S. NEWS`, `COMEDY`, etc.). If you later provide a dataset labeled `fake` / `real`, the same structure can be reused for true fake‑news detection.

---

## Project structure

- `train_model.py` – parses `News_clean.csv`, cleans text, trains multiple models, picks the best one and saves:
  - `fake_news_model.pkl`
  - `tfidf_vectorizer.pkl`
  - `label_encoder.pkl`
- `app.py` – Streamlit GUI that:
  - loads the saved model, TF‑IDF vectorizer and label encoder
  - applies the **same text cleaning** as training
  - lets you paste news text and shows the **predicted category**
- `predict.py` – simple command‑line script to test predictions.
- `News_clean.csv` – raw news data in a custom JSON‑like format.
- `requirements.txt` – Python dependencies for local use and Streamlit Cloud deployment.

All `.py`, `.pkl`, and `.txt` files are expected to live in the **same folder**.

---

## 1. Local setup and training

From the project directory:

```bash
pip install -r requirements.txt
```

Then train the model (this can take some time with the full dataset):

```bash
python train_model.py
```

After it finishes, you should see these new files in the folder:

- `fake_news_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

These are required by both `app.py` and `predict.py`.

---

## 2. Run the CLI predictor

From the same directory:

```bash
python predict.py
```

Then type or paste a news article, press Enter, and the script will print:

```text
Predicted category: <CATEGORY_NAME>
```

---

## 3. Run the Streamlit GUI locally

From the project directory:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`):

- Paste news text into the text box.
- Click **“Classify”**.
- The app will display **“Predicted category: <CATEGORY_NAME>”**.

---

## 4. Deploying on Streamlit Community Cloud

1. **Ensure your repository contains at least:**
   - `app.py`
   - `train_model.py`
   - `predict.py`
   - `requirements.txt`
   - `fake_news_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `label_encoder.pkl`

   (Generate the `.pkl` files locally by running `python train_model.py`, then commit them.)

2. **Push to GitHub** (for example, to a repo named `fake-news-detection`).

3. Go to `https://share.streamlit.io`, sign in with GitHub and click **“New app”**:
   - Repository: your repo (e.g. `username/fake-news-detection`)
   - Branch: `main` (or your default branch)
   - Main file path: `app.py`

4. Click **Deploy**.

Streamlit Cloud will:

- Install packages from `requirements.txt`.
- Run `app.py` from the repo root.
- Load the `.pkl` files from the same directory.

You will get a public URL for your complete, runnable project.

