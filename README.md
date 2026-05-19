# 🔍 Yelp Sentiment Analyser

</div>

| Name | Reg No | Course |
| --- | --- | --- |
| Uzma Haseeb | 253224 | MSc DataScience and BioAI|
| Anamika Ponnu | 253202 | M.Sc data Science and Bio AI |
|  | |  |

A Streamlit web app for sentiment analysis of Yelp reviews using TF-IDF features combined with handcrafted linguistic signals and four ML classifiers.A Streamlit-based web application for performing sentiment analysis on Yelp reviews using a hybrid feature engineering approach. The project combines TF-IDF vectorization with handcrafted linguistic features and evaluates multiple machine learning classifiers for accurate sentiment prediction.

---
# **Project Overview**

This project analyzes Yelp customer reviews and predicts whether a review expresses positive or negative sentiment.

The application:
Cleans and preprocesses review text
Extracts TF-IDF features
Generates handcrafted linguistic features
Trains and compares four machine learning models
Provides an interactive Streamlit interface for real-time predictions



## Features

- **Four classifiers** trained and compared side-by-side: Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Gradient Boosting
- **Rich feature set**: TF-IDF bigrams, lexicon scores, sentence length, average word length, punctuation density, and negation detection
- **Feature selection**: Chi-Square and Mutual Information rankings
- **Interactive results**: confusion matrices, per-class classification reports, top features per class
- **Live prediction**: single review analysis with word-level highlighting (positive, negative, negation words)
- **Batch prediction**: upload a CSV and download sentiment predictions

---

## Setup

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app-

```bash
streamlit run sentiment_app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Dataset Format-

Upload any CSV file with at least two columns:

| Column  | Description                        | Default name |
|---------|------------------------------------|--------------|
| `text`  | Raw review text                    | `text`       |
| `stars` | Numeric star rating (1 – 5)        | `stars`      |

Column names can be changed in the sidebar if yours differ.

**Label mapping applied internally:**

| Stars | Label    |
|-------|----------|
| 1 – 2 | Negative |
| 3     | Neutral  |
| 4 – 5 | Positive |

The [Yelp Open Dataset](https://www.yelp.com/dataset) (`yelp_academic_dataset_review.json`) is a good source. Convert it to CSV and select the `text` and `stars` columns before uploading.

---

## App Tabs-

| Tab        | What it does                                                                 |
|------------|------------------------------------------------------------------------------|
| 📂 Data    | Preview the uploaded CSV, view star-rating distribution and missing values   |
| 🚀 Train   | Train all four models with an 80/20 stratified split                         |
| 📊 Results | Compare F1 scores, inspect confusion matrices and top features per model     |
| 🔮 Predict | Analyse a single review or run batch predictions on a new CSV                |

---

## Requirements-

- Python 3.9 or later
- See `requirements.txt` for package versions

---

## Project Structure-

```
.
├── sentiment_app.py   # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---
 # Yelp Sentiment Analyser 📊-

A machine learning application to classify Yelp reviews using NLP techniques.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uzmahaseeb4-yelp-sentiment-analyser-app.streamlit.app)

---

## App Preview-

![App Deployment Screenshot](Screenshot%202026-05-14%20201253.png)

---

## Project Structure-

.
├── app.py              # Main Streamlit application
├── sentiment_model.py  # Model logic and feature extraction
├── requirements.txt    # Python dependencies
├── yelp.csv            # Dataset
└── README.md           # This file


---

## How It Works-

1. **Preprocessing** - Text is lowercased and cleaned.
2. **Feature Extraction** – TF-IDF bigrams are combined with handcrafted features.
3. **Feature Selection** – Chi-Square and Mutual Information rank the best tokens.
4. **Training** – Multiple classifiers are compared; the one with the highest macro F1 is used.
5. **Prediction** – Predicts sentiment with real-time word highlights.

 
