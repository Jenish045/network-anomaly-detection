import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(64, activation="relu")(input_layer)
    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dense(16, activation="relu")(encoded)
    encoded = Dense(8, activation="relu")(encoded)

    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(32, activation="relu")(decoded)
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="linear")(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    return autoencoder