from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["review"])

x_train_seq = tokenizer.texts_to_sequences(train_df["review"])
x_test_seq  = tokenizer.texts_to_sequences(test_df["review"])

maxlen = 200
x_train = pad_sequences(x_train_seq, maxlen=maxlen, padding='post')
x_test  = pad_sequences(x_test_seq,  maxlen=maxlen, padding='post')

y_train = train_df["label"].values
y_test  = test_df["label"].values
