import streamlit as st
import pickle
import string

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("models/fake_news_model.pkl", "rb") as f:
        return pickle.load(f)

model, tfidf = load_model()

# UI
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below to predict whether it's **REAL** or **FAKE**.")

text_input = st.text_area("📝 News Content", height=200)

if st.button("Predict"):
    cleaned = text_input.translate(str.maketrans('', '', string.punctuation)).lower()
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    st.success("✅ This news is REAL." if pred == 1 else "🚫 This news is FAKE.")
