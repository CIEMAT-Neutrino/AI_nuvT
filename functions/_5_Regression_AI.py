import numpy as np
import awkward as ak
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint



def split_train_test(image, hit_nuvT_filtered, test_ratio=0.50):
    """
    Split the image data and hit_nuvT_filtered into training and test sets.
    
    Parameters:
    - image_max_file: str, path to the image_max .npy file.
    - image_min_file: str, path to the image_min .npy file.
    - hit_nuvT_filtered: Array-like, the data to split.
    - test_ratio: float, the ratio of data to use for testing (default 0.50).
    
    Returns:
    - x_train: training set from image_max.
    - x_test: test set from image_max.
    - train_nuvT: training set from hit_nuvT_filtered.
    - test_nuvT: test set from hit_nuvT_filtered.
    """
    
    # Load the image_max and image_min files
    image = np.load(image)
    
    # Convert hit_nuvT_filtered to a numpy array

    hit_nuvT_filtered_flattened = ak.to_numpy(ak.flatten(hit_nuvT_filtered))
    
    # Ensure that the number of elements in image_min and hit_nuvT_filtered match
    assert np.shape(image)[0] == np.shape(hit_nuvT_filtered_flattened)[0], "Dimension mismatch between image and hit_nuvT_filtered_flattened"
    
    # Calculate the test size
    test_size = int(np.floor(test_ratio * np.shape(hit_nuvT_filtered_flattened)[0]))
    print(f"Test size: {test_size}")
    
    # Split image_max into training and test sets
    x_train, x_test = image[:-test_size], image[-test_size:]
    
    # Split hit_nuvT_filtered_flattened into training and test sets
    train_nuvT, test_nuvT = hit_nuvT_filtered_flattened[:-test_size], hit_nuvT_filtered_flattened[-test_size:]
    
    # Print shapes of the splits
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    print(f"train_nuvT length: {len(train_nuvT)}, test_nuvT length: {len(test_nuvT)}")
    
    return x_train, x_test, train_nuvT, test_nuvT


from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

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

    # Define the callbacks
    weights_file = "/tmp/weights_nuvT.hdf5.keras"
    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    callbacks = [checkpoint, early_stopping]

    # Train the model
    history = model.fit(
        x_train,
        train_nuvT,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(x_test, test_nuvT),
        verbose=1
    )

    # Load the best weights
    model.load_weights(weights_file)

    # Make predictions on the test set
    nuvT_pred = model.predict(x_test)

    return nuvT_pred, history
