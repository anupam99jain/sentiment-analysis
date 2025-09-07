import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- SETTINGS ----
MAXLEN = 300  # must match training

# ---- LOAD MODEL & TOKENIZER ----
model = load_model("sentiment_lstm.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ---- STREAMLIT UI ----
st.title("🎬 IMDB Movie Review Sentiment")
st.write("Type or paste a review below and click **Analyze**.")

user_text = st.text_area("Your review:", height=150)

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        seq = tokenizer.texts_to_sequences([user_text])
        padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')
        score = model.predict(padded)[0][0]
        sentiment = "🌟 Positive" if score > 0.5 else "💔 Negative"
        st.subheader(sentiment)
        st.caption(f"Raw score: {score:.3f}")
