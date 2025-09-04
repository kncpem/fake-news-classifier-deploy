# 📰 Fake News Classifier

This project classifies news articles as **Real** or **Fake** using various machine learning techniques. It includes implementations with **CountVectorizer**, **TF-IDF**, and **SBERT embeddings** combined with classifiers like **MultinomialNB**, **Passive Aggressive Classifier**, and **XGBoost**. The app is built with **Streamlit** for an interactive user interface.

---

## 🚀 How to Use

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Paste a news article in the app, adjust parameters (chunk size, threshold), and classify.

---

## 📊 Results

| **Model**                     | **Accuracy** | **ROC AUC** | **Notes**                     |
|-------------------------------|--------------|-------------|--------------------------------|
| CountVectorizer + MultinomialNB | 90.0%       | 0.94        | Bag-of-Words approach          |
| TF-IDF + Passive Aggressive    | 91.8%       | 0.95        | TF-IDF with linear classifier  |
| SBERT + XGBoost                | 86.4%       | 0.96        | Sentence embeddings + XGBoost  |

---

## 📂 Files

- `app.py`: Streamlit app for classification.
- `FakeNewsCount_vectorizer.ipynb`: CountVectorizer implementation.
- `FakeNewsClassifier TFIDF.ipynb`: TF-IDF implementation.
- `BERT.ipynb`: SBERT + XGBoost implementation.

---