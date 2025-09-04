import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# ---------------------------
# MUST be first Streamlit command
# ---------------------------
st.set_page_config(page_title="Fake News Classifier", layout="centered")

# ---------------------------
# Load Model & Embeddings
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_fake_news.pkl")   # your trained XGBoost
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  
    return model, embedder

xgb_clf, embedder = load_model()

# ---------------------------
# Helper Functions
# ---------------------------
def chunk_text(text, max_words=500):  # adjust size based on what you trained
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

def predict_fake_news(text):
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    preds = xgb_clf.predict(embeddings)

    # Majority vote
    final_pred = np.bincount(preds).argmax()
    return final_pred, preds

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📰 Fake News Classifier")
st.write("Enter text below and check if it's likely **Real or Fake News**")

user_input = st.text_area("Paste your news article here:", height=200)

if st.button("Classify"):
    if user_input.strip():
        final_pred, preds = predict_fake_news(user_input)
        label = "🟢 Real News" if final_pred == 0 else "🔴 Fake News"

        st.subheader("Final Prediction:")
        st.write(f"**{label}**")

        st.write("Chunk-wise predictions:")
        for i, p in enumerate(preds):
            lbl = "Real" if p == 0 else "Fake"
            st.write(f"Chunk {i+1}: {lbl}")
    else:
        st.warning("Please enter some text to classify.")