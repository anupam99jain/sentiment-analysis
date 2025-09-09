from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

maxlen = 200

model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_shape=(maxlen,)),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
