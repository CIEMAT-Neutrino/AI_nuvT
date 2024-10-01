import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

def create_cnn_model(x_train):
    """
    Creates a CNN model based on the provided architecture.
    
    Parameters:
    - input_shape: tuple, the shape of the input data (excluding batch size).
    
    Returns:
    - model: A compiled CNN model.
    """
    # Input layer
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    input_layer = layers.Input(shape=input_shape)
    
    # First convolutional block
    model = layers.BatchNormalization()(input_layer)
    model = layers.Conv2D(128, (3, 3), padding='same')(model)
    model = layers.LeakyReLU(alpha=0.1)(model)
    model = layers.MaxPooling2D((2, 2), padding='same')(model)
    model = layers.Dropout(0.3)(model)
    
    # Second convolutional block
    model = layers.Conv2D(256, (3, 3), padding='same')(model)
    model = layers.LeakyReLU(alpha=0.1)(model)
    model = layers.MaxPooling2D((2, 2), padding='same')(model)
    model = layers.Dropout(0.3)(model)
    
    # Third convolutional block
    model = layers.Conv2D(512, (3, 3), padding='same')(model)
    model = layers.LeakyReLU(alpha=0.1)(model)
    model = layers.MaxPooling2D((2, 2), padding='same')(model)
    model = layers.Dropout(0.4)(model)
    
    # Flatten the output and pass through dense layers
    model = layers.Flatten()(model)
    model = layers.Dense(2048, activation='relu')(model)
    model = layers.Dropout(0.2)(model)
    model = layers.Dense(512, activation='relu')(model)
    model = layers.Dense(128, activation='relu')(model)
    
    # Output layer
    output_layer = layers.Dense(1, activation='linear')(model)
    
    # Create the model
    model_def = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model_def.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model_def