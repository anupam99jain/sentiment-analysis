history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=128,
    verbose=1
)
