"""
=============================================================
  Sentiment Analysis of Yelp Reviews — Streamlit App
  TF-IDF + Linguistic Features + 4 ML Classifiers
=============================================================
  Run with:  streamlit run sentiment_app.py
=============================================================
"""

import os, re, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.feature_selection import chi2, mutual_info_classif

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Yelp Sentiment Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0f1117; color: #e8e6e1; }

[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #2a3040; }
[data-testid="stSidebar"] * { color: #c9c7c0 !important; }

h1 { font-family: 'DM Serif Display', serif !important; font-size: 2.4rem !important; color: #f0ece3 !important; letter-spacing: -0.5px; }
h2 { font-family: 'DM Serif Display', serif !important; font-size: 1.5rem !important; color: #d4c9b0 !important; }
h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; color: #b8b0a0 !important; font-size: 1rem !important; text-transform: uppercase; letter-spacing: 1.5px; }

[data-testid="metric-container"] { background: #1a2035; border: 1px solid #2a3550; border-radius: 8px; padding: 16px 20px; }
[data-testid="metric-container"] label { color: #8a9bb5 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; font-family: 'DM Mono', monospace !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e8e4da !important; font-family: 'DM Serif Display', serif !important; font-size: 2rem !important; }

[data-baseweb="tab-list"] { background: #161b27; border-bottom: 1px solid #2a3040; gap: 0; }
[data-baseweb="tab"] { color: #6b7a8d !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; padding: 12px 20px !important; border-radius: 0 !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #e8c97a !important; border-bottom: 2px solid #e8c97a !important; background: transparent !important; }

.stButton > button { background: #e8c97a; color: #0f1117; border: none; border-radius: 4px; font-family: 'DM Mono', monospace; font-size: 0.85rem; font-weight: 500; letter-spacing: 0.5px; padding: 10px 24px; transition: all 0.2s ease; }
.stButton > button:hover { background: #f0d88a; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(232,201,122,0.3); }

.stTextArea textarea, .stTextInput input { background: #1a2035 !important; border: 1px solid #2a3550 !important; color: #e8e4da !important; border-radius: 6px !important; font-family: 'DM Sans', sans-serif !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color: #e8c97a !important; box-shadow: 0 0 0 2px rgba(232,201,122,0.15) !important; }

[data-baseweb="select"] > div { background: #1a2035 !important; border-color: #2a3550 !important; color: #e8e4da !important; }

.pred-box { border-radius: 8px; padding: 28px 32px; margin: 16px 0; display: flex; align-items: center; gap: 20px; }
.pred-pos { background: linear-gradient(135deg, #0d2e1a, #143d24); border: 1px solid #2a6640; }
.pred-neg { background: linear-gradient(135deg, #2e0d0d, #3d1414); border: 1px solid #662a2a; }
.pred-neu { background: linear-gradient(135deg, #1e1e0d, #2e2d14); border: 1px solid #5a5a2a; }
.pred-label { font-family: 'DM Serif Display', serif; font-size: 2.2rem; }
.pred-sub   { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #8a9bb5; margin-top: 4px; }

.section-rule { border: none; border-top: 1px solid #2a3040; margin: 28px 0; }

.info-banner { background: #1a2035; border-left: 3px solid #e8c97a; border-radius: 0 6px 6px 0; padding: 12px 18px; font-family: 'DM Mono', monospace; font-size: 0.82rem; color: #b8b0a0; margin: 12px 0; }

.feat-pill { display: inline-block; background: #1a2035; border: 1px solid #2a3550; border-radius: 20px; padding: 4px 12px; font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #8ab4d4; margin: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SEED = 42

NEG_WORDS    = {'worst','bad','terrible','horrible','waste','disappointed',
                'return','broken','poor','hate','ugly','awful','crap','junk',
                'rude','slow','disgusting','dirty','overpriced','never'}
POS_WORDS    = {'great','excellent','love','best','perfect','amazing','good',
                'happy','awesome','nice','helpful','wonderful','like','fresh',
                'delicious','friendly','clean','fast','recommend','fantastic'}
NEGATION     = {'not','no','never','didnt','wasnt','cant','wont',
                'doesnt','neither','barely','nobody','nothing','without'}
INTENSIFIERS = {'very','extremely','really','so','totally','highly',
                'absolutely','incredibly','super','insanely'}

LABEL_COLORS = {'pos': '#4caf87', 'neg': '#e05c5c', 'neu': '#d4b84a'}
LABEL_EMOJI  = {'pos': '😊', 'neg': '😠', 'neu': '😐'}
LABEL_NAMES  = {'pos': 'POSITIVE', 'neg': 'NEGATIVE', 'neu': 'NEUTRAL'}


# ─────────────────────────────────────────────
# FEATURE FUNCTIONS
# ─────────────────────────────────────────────
def get_lexicon_features(text: str) -> list:
    words = str(text).lower().split()
    neg_score = pos_score = 0.0
    for i, word in enumerate(words):
        w   = re.sub(r"[^a-z]", "", word)
        ctx = [re.sub(r"[^a-z]", "", words[j]) for j in range(max(0, i-2), i)]
        is_neg = any(c in NEGATION     for c in ctx)
        is_int = any(c in INTENSIFIERS for c in ctx)
        mult   = 2.5 if is_int else 1.0
        if w in POS_WORDS:
            if is_neg: neg_score += 2.0 * mult
            else:      pos_score += 2.0 * mult
        if w in NEG_WORDS:
            if is_neg: pos_score += 2.0 * mult
            else:      neg_score += 2.0 * mult
    return [neg_score, pos_score]

def sentence_length(text):  return len(str(text).split())
def avg_word_length(text):
    w = str(text).split(); return np.mean([len(x) for x in w]) if w else 0.0
def punctuation_density(text):
    t = str(text); p = sum(1 for c in t if c in '!?.,;:')
    return p / len(t) if t else 0.0
def has_negation(text):
    words = set(re.sub(r"[^a-z\s]", "", str(text).lower()).split())
    return int(bool(words & NEGATION))
def clean_text(text):
    return re.sub(r"[^a-z\s]", " ", str(text).lower()).strip()

def extract_all_features(texts, tfidf_vect, lex_scaler=None, fit=False):
    if fit:
        X_tfidf = tfidf_vect.fit_transform(texts.apply(clean_text))
    else:
        X_tfidf = tfidf_vect.transform(texts.apply(clean_text))
    lex  = np.array([get_lexicon_features(t) for t in texts])
    slen = np.array([[sentence_length(t)]     for t in texts])
    awl  = np.array([[avg_word_length(t)]      for t in texts])
    pdns = np.array([[punctuation_density(t)]  for t in texts])
    hneg = np.array([[has_negation(t)]         for t in texts])
    hand = np.hstack([lex, slen, awl, pdns, hneg])
    if fit:
        lex_scaler  = MaxAbsScaler()
        hand_scaled = lex_scaler.fit_transform(hand)
    else:
        hand_scaled = lex_scaler.transform(hand)
    return hstack([X_tfidf, csr_matrix(hand_scaled)]), lex_scaler


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def run_training(df, text_col, star_col, progress_bar, status_text):
    df = df.dropna(subset=[text_col, star_col]).copy()
    df['label'] = df[star_col].apply(
        lambda x: 'neg' if float(x) <= 2 else ('neu' if float(x) == 3 else 'pos')
    )

    le    = LabelEncoder()
    y     = le.fit_transform(df['label'])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df[text_col], y, test_size=0.2, random_state=SEED, stratify=y
    )

    status_text.text("Extracting features…")
    progress_bar.progress(10)
    X_train, lex_scaler = extract_all_features(X_train_raw, tfidf, fit=True)
    X_test,  _          = extract_all_features(X_test_raw,  tfidf, lex_scaler=lex_scaler, fit=False)
    progress_bar.progress(25)

    # Feature selection
    status_text.text("Running feature selection…")
    X_tfidf_train = tfidf.transform(X_train_raw.apply(clean_text))
    chi2_scores, _ = chi2(X_tfidf_train, y_train)
    mi_scores       = mutual_info_classif(X_tfidf_train, y_train, random_state=SEED)
    feat_names      = np.array(tfidf.get_feature_names_out())
    chi2_top = feat_names[np.argsort(chi2_scores)[::-1][:15]].tolist()
    mi_top   = feat_names[np.argsort(mi_scores)[::-1][:15]].tolist()
    progress_bar.progress(40)

    mm         = MinMaxScaler(feature_range=(0, 1))
    X_train_nn = mm.fit_transform(X_train.toarray())
    X_test_nn  = mm.transform(X_test.toarray())

    model_configs = {
        "Logistic Regression":     LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=SEED),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM":              LinearSVC(class_weight='balanced', C=0.5, max_iter=2000, random_state=SEED),
        "Gradient Boosting":       HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=SEED),
    }

    reports = {}; f1_scores = {}; top_feats = {}
    best_f1 = -1; best_name = None; best_model = None

    for idx, (name, clf) in enumerate(model_configs.items()):
        status_text.text(f"Training {name}…")
        if name == "Multinomial Naive Bayes":
            clf.fit(X_train_nn, y_train);         y_pred = clf.predict(X_test_nn)
        elif name == "Gradient Boosting":
            clf.fit(X_train.toarray(), y_train);  y_pred = clf.predict(X_test.toarray())
        else:
            clf.fit(X_train, y_train);            y_pred = clf.predict(X_test)

        rep   = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        macro = f1_score(y_test, y_pred, average='macro')
        rep['_cm'] = confusion_matrix(y_test, y_pred)
        reports[name]   = rep
        f1_scores[name] = macro

        if macro > best_f1:
            best_f1 = macro; best_name = name; best_model = clf

        if hasattr(clf, 'coef_'):
            coef = clf.coef_
            if coef.shape[0] == 1: coef = np.vstack([-coef, coef])
            tf = {}
            for i, cls in enumerate(le.classes_):
                if i >= coef.shape[0]: break
                idx2 = np.argsort(coef[i])[::-1][:10]
                tf[cls] = feat_names[idx2[idx2 < len(feat_names)]].tolist()
            top_feats[name] = tf

        progress_bar.progress(40 + (idx + 1) * 14)

    progress_bar.progress(100)
    status_text.text("Done!")

    return {
        "model": best_model, "model_name": best_name,
        "tfidf": tfidf, "lex_scaler": lex_scaler, "mm_scaler": mm, "le": le,
        "f1_scores": f1_scores, "reports": reports, "top_feats": top_feats,
        "chi2_top": chi2_top, "mi_top": mi_top, "y_test": y_test,
        "label_dist": df['label'].value_counts().to_dict(),
        "n_train": len(y_train), "n_test": len(y_test),
    }


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict_review(text: str, pack: dict):
    texts = pd.Series([text])
    X, _  = extract_all_features(texts, pack['tfidf'], lex_scaler=pack['lex_scaler'], fit=False)
    name  = pack['model_name']
    if name == "Multinomial Naive Bayes":
        X = pack['mm_scaler'].transform(X.toarray())
    elif name == "Gradient Boosting":
        X = X.toarray()
    pred  = pack['model'].predict(X)[0]
    label = pack['le'].inverse_transform([pred])[0]
    return label, get_lexicon_features(text)


# ─────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────
PLOT_BG  = "#0f1117"
PLOT_PAP = "#161b27"
GRID_COL = "#2a3040"
FONT_COL = "#b8b0a0"

def base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(family="DM Serif Display", size=16, color="#d4c9b0")),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_PAP,
        font=dict(family="DM Mono", color=FONT_COL, size=11),
        xaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, linecolor=GRID_COL, zerolinecolor=GRID_COL),
        margin=dict(l=40, r=20, t=50, b=40),
    )

def bar_f1(f1_scores):
    names  = list(f1_scores.keys())
    scores = list(f1_scores.values())
    colors = ['#e8c97a' if s == max(scores) else '#2a3550' for s in scores]
    fig = go.Figure(go.Bar(
        x=names, y=scores, marker_color=colors,
        text=[f"{s:.4f}" for s in scores], textposition='outside',
        textfont=dict(family="DM Mono", size=11, color="#e8e4da"),
    ))
    fig.update_layout(**base_layout("Macro F1 — All Models"),
                      yaxis_range=[0, 1], showlegend=False)
    return fig

def confusion_heatmap(cm, classes):
    fig = go.Figure(go.Heatmap(
        z=cm, x=classes, y=classes,
        colorscale=[[0,'#161b27'],[0.5,'#1e3a5f'],[1,'#e8c97a']],
        text=cm, texttemplate="%{text}",
        textfont=dict(family="DM Mono", size=13),
        showscale=False,
    ))
    fig.update_layout(**base_layout("Confusion Matrix"),
                      xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def label_pie(label_dist):
    labels = list(label_dist.keys())
    vals   = list(label_dist.values())
    colors = [LABEL_COLORS.get(l, '#555') for l in labels]
    fig = go.Figure(go.Pie(
        labels=[LABEL_NAMES.get(l, l) for l in labels], values=vals,
        marker_colors=colors, hole=0.55,
        textfont=dict(family="DM Mono", size=11),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_PAP, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Mono", color=FONT_COL),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(font=dict(family="DM Mono")),
        title=dict(text="Label Distribution",
                   font=dict(family="DM Serif Display", size=15, color="#d4c9b0")),
    )
    return fig

def lex_bar(neg, pos):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Negative", x=["Score"], y=[neg],
                         marker_color='#e05c5c',
                         text=[f"{neg:.1f}"], textposition='outside'))
    fig.add_trace(go.Bar(name="Positive", x=["Score"], y=[pos],
                         marker_color='#4caf87',
                         text=[f"{pos:.1f}"], textposition='outside'))
    fig.update_layout(**base_layout("Lexicon Scores"),
                      barmode='group', yaxis_title="Score",
                      legend=dict(font=dict(family="DM Mono")))
    return fig

def stars_bar(df_raw, star_col):
    vc = df_raw[star_col].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[str(x) for x in vc.index], y=vc.values,
        marker_color='#e8c97a',
        text=vc.values, textposition='outside',
        textfont=dict(family="DM Mono", size=11, color="#e8e4da"),
    ))
    fig.update_layout(**base_layout("Star Rating Distribution"),
                      xaxis_title="Stars", yaxis_title="Count", showlegend=False)
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Yelp Sentiment")
    st.markdown("<div class='info-banner'>TF-IDF · LR · MNB · SVM · GB</div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Dataset")
    uploaded = st.file_uploader("Upload yelp.csv", type="csv",
                                label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Column names")
    text_col_opt = st.text_input("Text column",   value="text")
    star_col_opt = st.text_input("Rating column", value="stars")
    st.markdown("---")
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.72rem;"
        "color:#4a5568;line-height:2'>"
        "Split: 80 / 20 stratified<br>"
        "Features: TF-IDF bigrams<br>"
        "+ lexicon · sentence len<br>"
        "+ punct density · negation<br>"
        "Selection: χ² · Mutual Info"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("# Yelp Sentiment Analyser")
st.markdown(
    "<p style='color:#6b7a8d;font-family:DM Mono,monospace;font-size:0.85rem;"
    "margin-top:-12px'>TF-IDF · Chi-Square · Mutual Info · LR · MNB · SVM · Gradient Boosting</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

tab_data, tab_train, tab_results, tab_predict = st.tabs([
    "  📂  Data  ", "  🚀  Train  ", "  📊  Results  ", "  🔮  Predict  "
])

# ── TAB 1: DATA ──────────────────────────────
with tab_data:
    if uploaded is None:
        st.markdown(
            "<div class='info-banner'>Upload <code>yelp.csv</code> via the sidebar.<br>"
            "Expected columns: <code>text</code> (review) and <code>stars</code> (1–5 rating).</div>",
            unsafe_allow_html=True,
        )
    else:
        df_raw = pd.read_csv(uploaded)
        st.session_state['df_raw'] = df_raw

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",    f"{len(df_raw):,}")
        c2.metric("Columns",       len(df_raw.columns))
        c3.metric("Text column",   text_col_opt if text_col_opt in df_raw.columns else "⚠ missing")
        c4.metric("Rating column", star_col_opt if star_col_opt in df_raw.columns else "⚠ missing")

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1.4, 1])

        with col_l:
            st.markdown("### Preview")
            st.dataframe(df_raw.head(8), use_container_width=True)

        with col_r:
            if star_col_opt in df_raw.columns:
                st.plotly_chart(stars_bar(df_raw, star_col_opt), use_container_width=True)

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("### Missing values")
        miss = df_raw.isnull().sum().reset_index()
        miss.columns = ['Column', 'Missing']
        st.dataframe(miss, use_container_width=True, hide_index=True)


# ── TAB 2: TRAIN ─────────────────────────────
with tab_train:
    if 'df_raw' not in st.session_state:
        st.markdown("<div class='info-banner'>Upload data first (Data tab).</div>",
                    unsafe_allow_html=True)
    else:
        df_raw = st.session_state['df_raw']
        missing_cols = [c for c in [text_col_opt, star_col_opt]
                        if c not in df_raw.columns]
        if missing_cols:
            st.error(f"Column(s) not found: {missing_cols}. Available: {list(df_raw.columns)}")
        else:
            st.markdown("### Ready to train")
            st.markdown(
                f"<div class='info-banner'>"
                f"<b>{len(df_raw):,}</b> rows · text=<code>{text_col_opt}</code> · "
                f"rating=<code>{star_col_opt}</code> · 80/20 stratified split"
                f"</div>",
                unsafe_allow_html=True,
            )

            if st.button("🚀  Train All Models"):
                prog = st.progress(0)
                stat = st.empty()
                pack = run_training(df_raw, text_col_opt, star_col_opt, prog, stat)
                st.session_state['pack'] = pack
                stat.empty(); prog.empty()
                st.success(
                    f"✅  Done!  Best: **{pack['model_name']}**  "
                    f"(Macro F1 = {max(pack['f1_scores'].values()):.4f})"
                )
                cols = st.columns(4)
                for i, (name, score) in enumerate(pack['f1_scores'].items()):
                    short = (name.replace("Logistic Regression","Log. Reg.")
                                 .replace("Multinomial Naive Bayes","Naive Bayes")
                                 .replace("Gradient Boosting","Grad. Boost"))
                    delta = "best ✓" if name == pack['model_name'] else ""
                    cols[i].metric(short, f"{score:.4f}", delta)

            elif 'pack' in st.session_state:
                pack = st.session_state['pack']
                st.info(
                    f"Model already trained. Best: **{pack['model_name']}**  "
                    f"(Macro F1 = {max(pack['f1_scores'].values()):.4f})  — re-train anytime."
                )


# ── TAB 3: RESULTS ───────────────────────────
with tab_results:
    if 'pack' not in st.session_state:
        st.markdown("<div class='info-banner'>Train a model first.</div>",
                    unsafe_allow_html=True)
    else:
        pack = st.session_state['pack']

        r1, r2 = st.columns([1.6, 1])
        with r1:
            st.plotly_chart(bar_f1(pack['f1_scores']), use_container_width=True)
        with r2:
            st.plotly_chart(label_pie(pack['label_dist']), use_container_width=True)

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("### Per-model breakdown")
        chosen = st.selectbox("Model", list(pack['reports'].keys()),
                              label_visibility="collapsed")
        rep = pack['reports'][chosen]

        rows = []
        for cls in ['neg', 'neu', 'pos']:
            if cls in rep:
                r = rep[cls]
                rows.append({'Class': cls.upper(),
                             'Precision': f"{r['precision']:.3f}",
                             'Recall':    f"{r['recall']:.3f}",
                             'F1-score':  f"{r['f1-score']:.3f}",
                             'Support':   int(r['support'])})
        m = rep['macro avg']
        rows.append({'Class':'MACRO AVG',
                     'Precision': f"{m['precision']:.3f}",
                     'Recall':    f"{m['recall']:.3f}",
                     'F1-score':  f"{m['f1-score']:.3f}",
                     'Support':   int(m['support'])})

        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c1:
            st.markdown("**Classification Report**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with c2:
            st.plotly_chart(confusion_heatmap(rep['_cm'], ['neg','neu','pos']),
                            use_container_width=True)
        with c3:
            if chosen in pack['top_feats']:
                st.markdown("**Top Features per Class**")
                for cls, feats in pack['top_feats'][chosen].items():
                    st.markdown(
                        f"<div style='color:{LABEL_COLORS[cls]};font-family:DM Mono,monospace;"
                        f"font-size:0.78rem;margin-top:8px'>{LABEL_NAMES[cls]}</div>",
                        unsafe_allow_html=True,
                    )
                    pills = "".join(f"<span class='feat-pill'>{f}</span>" for f in feats)
                    st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='info-banner'>Feature coefficients not available for this model.</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("### Feature Selection")
        fs1, fs2 = st.columns(2)
        with fs1:
            st.markdown("**Chi-Square — Top 15**")
            pills = "".join(f"<span class='feat-pill'>{f}</span>" for f in pack['chi2_top'])
            st.markdown(pills, unsafe_allow_html=True)
        with fs2:
            st.markdown("**Mutual Information — Top 15**")
            pills = "".join(f"<span class='feat-pill'>{f}</span>" for f in pack['mi_top'])
            st.markdown(pills, unsafe_allow_html=True)


# ── TAB 4: PREDICT ───────────────────────────
with tab_predict:
    if 'pack' not in st.session_state:
        st.markdown("<div class='info-banner'>Train a model first.</div>",
                    unsafe_allow_html=True)
    else:
        pack = st.session_state['pack']
        best = pack['model_name']

        st.markdown(
            f"### Analyse a review "
            f"<span style='font-family:DM Mono;font-size:0.78rem;color:#6b7a8d'>"
            f"using {best}</span>",
            unsafe_allow_html=True,
        )

        review_input = st.text_area(
            "Review",
            placeholder="e.g.  The food was absolutely amazing — highly recommend!",
            height=120,
            label_visibility="collapsed",
        )

        if st.button("  🔮  Analyse Sentiment  "):
            if review_input.strip():
                label, lex = predict_review(review_input, pack)
                col_hex    = LABEL_COLORS[label]
                css_cls    = f"pred-{label}"

                st.markdown(
                    f"<div class='pred-box {css_cls}'>"
                    f"  <div style='font-size:3rem'>{LABEL_EMOJI[label]}</div>"
                    f"  <div>"
                    f"    <div class='pred-label' style='color:{col_hex}'>{LABEL_NAMES[label]}</div>"
                    f"    <div class='pred-sub'>Model: {best} · "
                    f"lexicon neg={lex[0]:.1f}  pos={lex[1]:.1f}</div>"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                p_left, p_right = st.columns([1, 1.6])
                with p_left:
                    st.plotly_chart(lex_bar(lex[0], lex[1]), use_container_width=True)

                with p_right:
                    st.markdown("**Word highlights**")
                    words = review_input.split()
                    highlighted = []
                    for w in words:
                        cw = re.sub(r"[^a-z]", "", w.lower())
                        if cw in POS_WORDS:
                            highlighted.append(
                                f"<span style='background:#143d24;color:#4caf87;"
                                f"border-radius:3px;padding:1px 5px'>{w}</span>")
                        elif cw in NEG_WORDS:
                            highlighted.append(
                                f"<span style='background:#3d1414;color:#e05c5c;"
                                f"border-radius:3px;padding:1px 5px'>{w}</span>")
                        elif cw in NEGATION:
                            highlighted.append(
                                f"<span style='background:#2e2d14;color:#d4b84a;"
                                f"border-radius:3px;padding:1px 5px'>{w}</span>")
                        else:
                            highlighted.append(
                                f"<span style='color:#b8b0a0'>{w}</span>")

                    st.markdown(
                        "<div style='background:#1a2035;border:1px solid #2a3550;"
                        "border-radius:8px;padding:16px 20px;font-family:DM Sans,sans-serif;"
                        "font-size:0.95rem;line-height:2;word-spacing:4px'>"
                        + " ".join(highlighted) + "</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='font-family:DM Mono,monospace;font-size:0.72rem;"
                        "color:#4a5568;margin-top:8px'>"
                        "<span style='background:#143d24;color:#4caf87;padding:2px 7px;"
                        "border-radius:3px'>pos word</span> &nbsp;"
                        "<span style='background:#3d1414;color:#e05c5c;padding:2px 7px;"
                        "border-radius:3px'>neg word</span> &nbsp;"
                        "<span style='background:#2e2d14;color:#d4b84a;padding:2px 7px;"
                        "border-radius:3px'>negation</span></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Please enter a review first.")

        # ── Batch predict ─────────────────────
        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("### Batch Predict from CSV")
        batch_file = st.file_uploader("Upload CSV", type="csv", key="batch",
                                      label_visibility="collapsed")
        if batch_file:
            bdf  = pd.read_csv(batch_file)
            bcol = st.selectbox("Text column to predict", bdf.columns, key="bcol")
            if st.button("  Run Batch Predict  "):
                preds = [predict_review(str(t), pack)[0]
                         for t in bdf[bcol].fillna("")]
                bdf['sentiment'] = preds
                st.dataframe(bdf[[bcol, 'sentiment']].head(30),
                             use_container_width=True)
                st.download_button(
                    "⬇  Download predictions.csv",
                    bdf.to_csv(index=False).encode('utf-8'),
                    file_name="predictions.csv", mime="text/csv",
                )
