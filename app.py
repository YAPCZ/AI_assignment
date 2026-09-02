#command for installing dependencies
#pip install streamlit joblib torch nltk transformers scikit-learn

import streamlit as st
import joblib
import re
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# Download NLTK data (only runs once, then cached)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Load Naive Bayes model + vectorizer (cached so it only loads once) ---
@st.cache_resource
def load_nb_model():
    model = joblib.load('nb_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# --- Load DistilBERT pipeline (cached so it only downloads/loads once) ---
@st.cache_resource
def load_distilbert():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        'sentiment-analysis',
        model='distilbert-base-uncased-finetuned-sst-2-english',  # or 'distilbert_sentiment' if you saved it locally
        device=device,
        truncation=True,
        max_length=512
    )

nb_model, tfidf = load_nb_model()

# --- Same preprocessing function used during training ---
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- UI ---
st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and choose which model predicts whether it's positive or negative.")

model_choice = st.radio("Choose a model", ["Naive Bayes (TF-IDF)", "DistilBERT (Transformer)"])

review = st.text_area("Movie review", height=150, placeholder="Type or paste a movie review here...")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")

    elif model_choice == "Naive Bayes (TF-IDF)":
        cleaned = preprocess_text(review)
        vec = tfidf.transform([cleaned])
        prediction = nb_model.predict(vec)[0]
        proba = nb_model.predict_proba(vec)[0]

        if prediction == 'positive':
            st.success(f"**Positive** 😀 (confidence: {max(proba)*100:.1f}%)")
        else:
            st.error(f"**Negative** 😞 (confidence: {max(proba)*100:.1f}%)")

        with st.expander("See cleaned/preprocessed text"):
            st.write(cleaned)

    else:  # DistilBERT
        with st.spinner("Loading DistilBERT (first run only)..."):
            sentiment_pipeline = load_distilbert()
        result = sentiment_pipeline(review)[0]

        if result['label'] == 'POSITIVE':
            st.success(f"**Positive** 😀 (confidence: {result['score']*100:.1f}%)")
        else:
            st.error(f"**Negative** 😞 (confidence: {result['score']*100:.1f}%)")

st.markdown("---")
st.caption("Naive Bayes: TF-IDF + MultinomialNB, trained on the IMDB dataset. "
           "DistilBERT: pretrained transformer loaded from Hugging Face Hub.")
