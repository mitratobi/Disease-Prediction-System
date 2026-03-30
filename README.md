# Disease Prediction System

A machine learning web app that predicts diseases based on symptoms using a Random Forest classifier trained on 41 diseases and 95 symptoms.

## Project Structure

```
disease-prediction/
│
├── data/
│   ├── Training.csv
│   └── Testing.csv
│
├── model/
│   ├── random_forest_model.pkl
│   └── model_meta.pkl
│
├── src/
│   ├── preprocess.py     ← data cleaning & feature selection
│   ├── train.py          ← model training & evaluation
│   └── predict.py        ← load model & predict from symptoms
│
├── streamlit_app.py      ← web UI (Streamlit)
├── app.py                ← CLI interface
└── requirements.txt
```

## How It Works

1. `preprocess.py` loads the CSV, selects the top 95 symptoms by frequency, and encodes disease labels
2. `train.py` trains a `RandomForestClassifier` (100 estimators), evaluates on test data (~97% accuracy), and saves the model
3. `predict.py` takes a list of symptoms, builds a binary input vector, and returns the predicted disease
4. `streamlit_app.py` provides a searchable dropdown UI for symptom selection

---

## Local Setup

### 1. Clone the repository

```bash
git clone git@github.com:mitratobi/Disease-Prediction-System.git
cd Disease-Prediction-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

### 4. Run the web app

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

### 4a. Run the CLI instead (optional)

```bash
python app.py
```

---

## Deployment (Streamlit Community Cloud)

Free hosting directly from your GitHub repo — no server setup needed.

### Steps

1. Push your code to GitHub (already done if you cloned this repo)

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account

3. Click **"New app"** and fill in:
   - **Repository:** `mitratobi/Disease-Prediction-System`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`

4. Click **Deploy**

Streamlit installs dependencies from `requirements.txt` automatically and provides a public URL (e.g. `mitratobi-disease-prediction.streamlit.app`).

### Updating the deployed app

Every push to `main` triggers an automatic redeploy:

```bash
git add .
git commit -m "your message"
git push
```

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest (n=100) |
| Training samples | 4,920 |
| Test samples | 42 |
| Test accuracy | 97.62% |
| Diseases covered | 41 |
| Symptoms used | 95 |

---

> **Disclaimer:** This tool is for educational purposes only. Always consult a qualified medical professional for diagnosis.
