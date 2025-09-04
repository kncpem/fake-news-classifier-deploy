import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import re

# ----------------------------
# Page config MUST be first
# ----------------------------
st.set_page_config(page_title="📰 Fake News Classifier", layout="centered")

# ----------------------------
# Settings
# ----------------------------
# Adjust to your training choice. Most of our earlier code used 0=Real, 1=Fake.
LABEL_TEXT = {0: "🟢 Real News", 1: "🔴 Fake News"}
CHUNK_MAX_WORDS = 400  # keep consistent with how you trained

# ----------------------------
# Cache heavy objects
# ----------------------------
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Load your pickled XGBoost model
    xgb_clf = joblib.load("xgb_fake_news.pkl")
    return embedder, xgb_clf

embedder, xgb_clf = load_resources()

# ----------------------------
# Helpers
# ----------------------------
def split_text(text: str, max_words: int = CHUNK_MAX_WORDS):
    words = re.split(r"\s+", text.strip())
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words) if words[i:i+max_words]]

@st.cache_data
def embed_chunks(chunks):
    # returns a numpy array (n_chunks, dim)
    return embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

def predict_article(text: str):
    chunks = split_text(text)
    if not chunks:
        return None, None, None

    X = embed_chunks(chunks)

    # Prefer probability averaging if available (more stable than hard vote)
    if hasattr(xgb_clf, "predict_proba"):
        proba = xgb_clf.predict_proba(X)[:, 1]  # probability of label "1" (Fake)
        mean_p_fake = float(np.mean(proba))
        final_label = 1 if mean_p_fake >= 0.5 else 0
        return final_label, proba, chunks
    else:
        preds = xgb_clf.predict(X)
        # majority vote fallback
        vals, counts = np.unique(preds, return_counts=True)
        final_label = int(vals[np.argmax(counts)])
        return final_label, preds, chunks

# ----------------------------
# UI
# ----------------------------
st.title("📰 Fake News Classifier (SBERT + XGBoost)")
st.write("Paste an article; long inputs are automatically chunked and aggregated.")

user_text = st.text_area("Article text", height=220, placeholder="Paste news article here...")

col1, col2 = st.columns([1, 1])
with col1:
    chunk_size = st.number_input("Chunk size (words)", min_value=100, max_value=800, value=CHUNK_MAX_WORDS, step=50)
with col2:
    thresh = st.slider("Fake threshold (probability)", 0.05, 0.95, 0.50, 0.05)

if st.button("Classify"):
    if not user_text.strip():
        st.warning("Please paste some text.")
    else:
        # temporarily override chunk size if changed by user
        # global CHUNK_MAX_WORDS
        # CHUNK_MAX_WORDS = int(chunk_size)

        label, scores, chunks = predict_article(user_text)
        if label is None:
            st.error("Could not process input.")
        else:
            # If we had probabilities, recalc with user-chosen threshold for display
            if scores is not None and scores.ndim == 1 and scores.size == len(chunks):
                mean_p_fake = float(np.mean(scores))
                label = 1 if mean_p_fake >= thresh else 0
                st.subheader(f"Final: **{LABEL_TEXT[label]}**")
                st.caption(f"Mean P(Fake) across chunks: {mean_p_fake:.3f}  •  threshold={thresh:.2f}")
                with st.expander("Chunk details"):
                    for i, (c, p) in enumerate(zip(chunks, scores), start=1):
                        st.write(f"**Chunk {i}** — P(Fake)={p:.3f}")
                        st.write(c[:300] + ("…" if len(c) > 300 else ""))
                        st.markdown("---")
            else:
                st.subheader(f"Final: **{LABEL_TEXT[label]}**")
                with st.expander("Chunk predictions"):
                    for i, (c, p) in enumerate(zip(chunks, scores), start=1):
                        st.write(f"**Chunk {i}** — {LABEL_TEXT[int(p)]}")
                        st.write(c[:300] + ("…" if len(c) > 300 else ""))
                        st.markdown("---")