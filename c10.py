MAX_WORDS = 20000
MAXLEN = 300
sample = "The movie was absolutely fantastic, brilliant plot and acting."
seq = tokenizer.texts_to_sequences([sample])
padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')
prediction = model.predict(padded)
print("Positive" if prediction[0][0] > 0.5 else "Negative", prediction)
