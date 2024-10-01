from tensorflow.keras.callbacks import ModelCheckpoint

def train_and_predict(model, x_train, train_nuvT, x_test, test_nuvT, epochs=30, batch_size=32):
    """
    Trains the model and predicts the output.

    Parameters:
        model: The Keras model to be trained.
        x_train: Training input data.
        train_nuvT: Training output data.
        x_test: Testing input data.
        test_nuvT: Testing output data.
        epochs: Number of epochs for training.
        batch_size: Batch size for training.
        weights_file: Path to save the best weights.

    Returns:
        nuvT_pred: Predictions on the test data.
    """

    # Define the ModelCheckpoint callback
    weights_file="/tmp/weights_nuvT.hdf5"
    checkpoint = ModelCheckpoint(weights_file, monitor='mse', verbose=0, save_best_only=True, mode='min')
    callbacks = [checkpoint]

    # Train the model
    history = model.fit(
        x_train,
        train_nuvT,
        epochs=epochs,
        batch_size=batch_size,  # Updated parameter name to 'batch_size'
        callbacks=callbacks,
        validation_data=(x_test, test_nuvT),
        verbose=2
    )

    # Load the best weights
    model.load_weights(weights_file)

    # Make predictions on the test set
    nuvT_pred = model.predict(x_test)

    return nuvT_pred, history
