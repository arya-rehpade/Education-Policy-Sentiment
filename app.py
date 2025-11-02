# app.py
import streamlit as st
import joblib, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Sentiment Model Comparison", page_icon="🎯")

# -----------------------
# Helpers
# -----------------------
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_resource
def load_assets():
    assets = {}
    assets['tfidf'] = joblib.load("tfidf_vectorizer.joblib")
    assets['LogisticRegression'] = joblib.load("LogisticRegression_model.joblib")
    # SVM might be LinearSVC without predict_proba, but predict works
    try:
        assets['SVM'] = joblib.load("SVM_model.joblib")
    except:
        assets['SVM'] = None
    try:
        assets['NaiveBayes'] = joblib.load("NaiveBayes_model.joblib")
    except:
        assets['NaiveBayes'] = None
    try:
        assets['results_df'] = pd.read_csv("model_comparison_results.csv")
    except:
        assets['results_df'] = None
    return assets

assets = load_assets()
tfidf = assets['tfidf']
available_models = [k for k in ['LogisticRegression','SVM','NaiveBayes'] if assets.get(k) is not None]

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Controls")
mode = st.sidebar.radio("Mode:", ("Single input", "Bulk CSV"))
sel_models = st.sidebar.multiselect("Pick model(s):", options=available_models, default=available_models)
show_metrics = st.sidebar.checkbox("Show model metrics", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Upload models (.joblib) or model_comparison_results.csv to update app")

# -----------------------
# Layout
# -----------------------
st.title("🎓 Education Policy Sentiment — Model Comparison")
col_left, col_right = st.columns([2,1])

with col_left:
    st.header("Prediction")
    if mode == "Single input":
        text = st.text_area("Enter text to analyze:", height=140)
        if st.button("Predict"):
            if not text.strip():
                st.warning("Enter some text")
            else:
                txt = clean_text(text)
                vec = tfidf.transform([txt])
                results = {}
                for m in sel_models:
                    model = assets.get(m)
                    if model is None:
                        results[m] = "model-not-found"
                    else:
                        pred = model.predict(vec)[0]
                        results[m] = pred
                # Display side-by-side
                cols = st.columns(len(results))
                for i,(m,p) in enumerate(results.items()):
                    with cols[i]:
                        st.subheader(m)
                        if p == "positive":
                            st.success("✅ Positive")
                        elif p == "negative":
                            st.error("❌ Negative")
                        elif p == "neutral":
                            st.info("😐 Neutral")
                        else:
                            st.write(p)
    else:
        st.header("Bulk CSV predictions")
        uploaded = st.file_uploader("Upload CSV (must contain text column)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Columns:", list(df.columns))
            text_col = st.selectbox("Select text column", df.columns)
            if st.button("Run bulk predictions"):
                df[text_col] = df[text_col].astype(str)
                df['clean_text'] = df[text_col].apply(clean_text)
                X = tfidf.transform(df['clean_text'].tolist())
                for m in sel_models:
                    model = assets.get(m)
                    if model is None:
                        df[f'pred_{m}'] = "model-not-found"
                    else:
                        df[f'pred_{m}'] = list(model.predict(X))
                st.dataframe(df.head(200))
                csv = df.to_csv(index=False).encode()
                st.download_button("Download predictions CSV", csv, file_name="predictions.csv")

with col_right:
    st.header("Model Info")
    st.write("Loaded models:", ", ".join(available_models))
    if assets.get('results_df') is not None and show_metrics:
        st.markdown("### Comparison Table")
        st.dataframe(assets['results_df'])
        st.markdown("### F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(data=assets['results_df'], x='Model', y='F1_macro', ax=ax)
        ax.set_ylim(0,1)
        st.pyplot(fig)
    else:
        st.info("No `model_comparison_results.csv` found (optional).")

st.markdown("---")
st.caption("App uses TF-IDF + saved classical ML models.")
