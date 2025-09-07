%%writefile app.py
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 300

# Load model & tokenizer
model = load_model("sentiment_lstm.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title("🎬 IMDB Sentiment Analysis")
st.write("Type a movie review and click **Analyze**")

review = st.text_area("Your Review:", height=150)

if st.button("Analyze"):
    if review.strip():
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAXLEN, padding="post")
        score = model.predict(padded)[0][0]
        sentiment = "🌟 Positive" if score > 0.5 else "💔 Negative"
        st.subheader(sentiment)
        st.caption(f"Confidence: {score:.3f}")
    else:
        st.warning("Please enter some text.")
