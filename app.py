import streamlit as st
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Load your trained model and CountVectorizer
model = joblib.load('fake_news_classifier.pkl')
cv = joblib.load('count_vectorizer.pkl')

nltk.download('stopwords')

st.title('Fake News Classifier')

input_text = st.text_area("Enter the news text:")

def preprocess_title(title):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', title)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

if st.button("Classify"):
    preprocessed_title = preprocess_title(input_text)
    
    # Transform the input text using the fitted CountVectorizer
    title_vector = cv.transform([preprocessed_title]).toarray()
    
    # Perform prediction
    prediction = model.predict(title_vector)
    
    try:
        prediction_proba = model._predict_proba_lr(title_vector)
        confidence = prediction_proba[0][prediction[0]] * 100
    except AttributeError:
        confidence = "N/A"

    # Output using Streamlit
    st.write(f"Prediction: {'Real News' if prediction[0] == 1 else 'Fake News'}")
    st.write(f"Confidence: {confidence if isinstance(confidence, str) else f'{confidence:.2f}%'}")