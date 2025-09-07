score = model.predict(padded)[0][0]
st.write("DEBUG Score:", score)  # <-- add this
label = "🌟 Positive" if score > 0.5 else "💔 Negative"
