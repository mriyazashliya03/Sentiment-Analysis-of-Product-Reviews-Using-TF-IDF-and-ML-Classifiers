# ============================================================
# YELP REVIEW SENTIMENT ANALYSIS - MACHINE LEARNING PIPELINE
# ============================================================

import re
import warnings
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler
)

from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix
)

from sklearn.feature_selection import (
    chi2,
    mutual_info_classif
)

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================

SEED = 42

NEG_WORDS = {
    'worst', 'bad', 'terrible', 'horrible', 'waste',
    'disappointed', 'return', 'broken', 'poor',
    'hate', 'ugly', 'awful', 'crap', 'junk',
    'rude', 'slow', 'disgusting', 'dirty',
    'overpriced', 'never'
}

POS_WORDS = {
    'great', 'excellent', 'love', 'best', 'perfect',
    'amazing', 'good', 'happy', 'awesome',
    'nice', 'helpful', 'wonderful', 'like',
    'fresh', 'delicious', 'friendly', 'clean',
    'fast', 'recommend', 'fantastic'
}

NEGATION = {
    'not', 'no', 'never', 'didnt', 'wasnt',
    'cant', 'wont', 'doesnt', 'neither',
    'barely', 'nobody', 'nothing', 'without'
}

INTENSIFIERS = {
    'very', 'extremely', 'really', 'so',
    'totally', 'highly', 'absolutely',
    'incredibly', 'super', 'insanely'
}

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean review text by:
    - converting to lowercase
    - removing special characters
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()


# ============================================================
# HANDCRAFTED LEXICON FEATURES
# ============================================================

def get_lexicon_features(text):
    """
    Extract positive and negative sentiment scores.
    """

    words = str(text).lower().split()

    neg_score = 0.0
    pos_score = 0.0

    for i, word in enumerate(words):

        current_word = re.sub(r"[^a-z]", "", word)

        context = [
            re.sub(r"[^a-z]", "", words[j])
            for j in range(max(0, i - 2), i)
        ]

        has_negation = any(w in NEGATION for w in context)
        has_intensifier = any(w in INTENSIFIERS for w in context)

        multiplier = 2.5 if has_intensifier else 1.0

        # Positive words
        if current_word in POS_WORDS:

            if has_negation:
                neg_score += 2.0 * multiplier
            else:
                pos_score += 2.0 * multiplier

        # Negative words
        if current_word in NEG_WORDS:

            if has_negation:
                pos_score += 2.0 * multiplier
            else:
                neg_score += 2.0 * multiplier

    return [neg_score, pos_score]


# ============================================================
# ADDITIONAL FEATURES
# ============================================================

def sentence_length(text):
    return len(str(text).split())


def avg_word_length(text):

    words = str(text).split()

    if len(words) == 0:
        return 0.0

    return np.mean([len(word) for word in words])


def punctuation_density(text):

    text = str(text)

    punct_count = sum(1 for c in text if c in '!?.,;:')

    if len(text) == 0:
        return 0.0

    return punct_count / len(text)


def has_negation(text):

    words = set(
        re.sub(r"[^a-z\s]", "", str(text).lower()).split()
    )

    return int(bool(words & NEGATION))


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_all_features(
    texts,
    tfidf_vectorizer,
    lex_scaler=None,
    fit=False
):
    """
    Combine:
    - TF-IDF features
    - Lexicon features
    - Linguistic features
    """

    # TF-IDF
    if fit:
        X_tfidf = tfidf_vectorizer.fit_transform(
            texts.apply(clean_text)
        )
    else:
        X_tfidf = tfidf_vectorizer.transform(
            texts.apply(clean_text)
        )

    # Lexicon features
    lex_features = np.array([
        get_lexicon_features(text)
        for text in texts
    ])

    # Linguistic features
    sentence_len = np.array([
        [sentence_length(text)]
        for text in texts
    ])

    avg_len = np.array([
        [avg_word_length(text)]
        for text in texts
    ])

    punct_density = np.array([
        [punctuation_density(text)]
        for text in texts
    ])

    negation = np.array([
        [has_negation(text)]
        for text in texts
    ])

    handcrafted_features = np.hstack([
        lex_features,
        sentence_len,
        avg_len,
        punct_density,
        negation
    ])

    # Scale handcrafted features
    if fit:
        lex_scaler = MaxAbsScaler()
        handcrafted_scaled = lex_scaler.fit_transform(
            handcrafted_features
        )
    else:
        handcrafted_scaled = lex_scaler.transform(
            handcrafted_features
        )

    # Combine TF-IDF + handcrafted features
    final_features = hstack([
        X_tfidf,
        csr_matrix(handcrafted_scaled)
    ])

    return final_features, lex_scaler


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_models(df, text_column="text", rating_column="stars"):

    # --------------------------------------------------------
    # LABEL CREATION
    # --------------------------------------------------------

    df = df.dropna(subset=[text_column, rating_column]).copy()

    df["label"] = df[rating_column].apply(
        lambda x:
            "neg" if float(x) <= 2
            else ("neu" if float(x) == 3 else "pos")
    )

    # Encode labels
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(df["label"])

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[text_column],
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    X_train, lex_scaler = extract_all_features(
        X_train_raw,
        tfidf,
        fit=True
    )

    X_test, _ = extract_all_features(
        X_test_raw,
        tfidf,
        lex_scaler=lex_scaler,
        fit=False
    )

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    X_tfidf_train = tfidf.transform(
        X_train_raw.apply(clean_text)
    )

    chi2_scores, _ = chi2(X_tfidf_train, y_train)

    mi_scores = mutual_info_classif(
        X_tfidf_train,
        y_train,
        random_state=SEED
    )

    feature_names = np.array(
        tfidf.get_feature_names_out()
    )

    chi2_top_features = feature_names[
        np.argsort(chi2_scores)[::-1][:15]
    ]

    mi_top_features = feature_names[
        np.argsort(mi_scores)[::-1][:15]
    ]

    # --------------------------------------------------------
    # SCALING FOR NAIVE BAYES
    # --------------------------------------------------------

    minmax_scaler = MinMaxScaler()

    X_train_nb = minmax_scaler.fit_transform(
        X_train.toarray()
    )

    X_test_nb = minmax_scaler.transform(
        X_test.toarray()
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                random_state=SEED
            ),

        "Multinomial Naive Bayes":
            MultinomialNB(alpha=0.5),

        "Linear SVM":
            LinearSVC(
                class_weight="balanced",
                C=0.5,
                max_iter=2000,
                random_state=SEED
            ),

        "Gradient Boosting":
            HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                random_state=SEED
            )
    }

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    reports = {}
    f1_scores = {}

    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        # Naive Bayes
        if model_name == "Multinomial Naive Bayes":

            model.fit(X_train_nb, y_train)

            predictions = model.predict(X_test_nb)

        # Gradient Boosting
        elif model_name == "Gradient Boosting":

            model.fit(X_train.toarray(), y_train)

            predictions = model.predict(X_test.toarray())

        # Sparse matrix models
        else:

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

        # Evaluation
        report = classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            output_dict=True
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        confusion = confusion_matrix(
            y_test,
            predictions
        )

        reports[model_name] = {
            "classification_report": report,
            "confusion_matrix": confusion
        }

        f1_scores[model_name] = macro_f1

        print(f"Macro F1 Score: {macro_f1:.4f}")

        # Best model tracking
        if macro_f1 > best_f1:

            best_f1 = macro_f1
            best_model = model
            best_model_name = model_name

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n================================================")
    print(f"BEST MODEL: {best_model_name}")
    print(f"BEST MACRO F1: {best_f1:.4f}")
    print("================================================")

    return {

        "best_model": best_model,
        "best_model_name": best_model_name,

        "tfidf_vectorizer": tfidf,
        "lex_scaler": lex_scaler,
        "minmax_scaler": minmax_scaler,

        "label_encoder": label_encoder,

        "f1_scores": f1_scores,
        "reports": reports,

        "chi2_top_features": chi2_top_features,
        "mi_top_features": mi_top_features
    }


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(review_text, model_package):

    texts = pd.Series([review_text])

    X, _ = extract_all_features(
        texts,
        model_package["tfidf_vectorizer"],
        lex_scaler=model_package["lex_scaler"],
        fit=False
    )

    model_name = model_package["best_model_name"]

    # Naive Bayes
    if model_name == "Multinomial Naive Bayes":

        X = model_package["minmax_scaler"].transform(
            X.toarray()
        )

    # Gradient Boosting
    elif model_name == "Gradient Boosting":

        X = X.toarray()

    prediction = model_package["best_model"].predict(X)[0]

    label = model_package["label_encoder"].inverse_transform(
        [prediction]
    )[0]

    return label


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("yelp.csv")

    # Train models
    model_package = train_models(
        df,
        text_column="text",
        rating_column="stars"
    )

    # Test prediction
    sample_review = "The food was amazing and the staff was very friendly."

    sentiment = predict_sentiment(
        sample_review,
        model_package
    )

    print("\nSample Review:")
    print(sample_review)

    print("\nPredicted Sentiment:")
    print(sentiment)