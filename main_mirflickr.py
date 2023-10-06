import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers, models


image_height = 300
image_width = 300




def get_model(shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=shape))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    # model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    # # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPool2D(2, 2))
    # model.add(layers.Dropout(0.3))

    # model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPool2D(2, 2))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # model.add(layers.Dense(128, activation="relu"))
    # # model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.3))

    model.add(layers.Dense(25, activation="sigmoid"))

    
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    model = get_model(X_train[0].shape)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    # history = model.fit(X_train, y_train, epochs=15, callbacks=[es_callback], validation_split=0.3, batch_size=16)
    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)


