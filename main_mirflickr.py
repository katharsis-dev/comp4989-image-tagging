import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from generator import Generator
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.resnet import ResNet152, ResNet50
from tensorflow.keras.models import Model


image_height = 300
image_width = 300

def get_model(shape, output):
    # Create a Sequential model
    model = tf.keras.Sequential()

    # First Convolutional Layer
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=shape, padding='valid', activation='relu'))

    # Second Convolutional Layer
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))

    # Third Convolutional Layer
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    # Fourth Convolutional Layer
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))

    # Fifth Convolutional Layer
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    # Flatten the output for fully connected layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Fully Connected Layer 2
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer (assuming a specific number of classes)
    model.add(Dense(output, activation='sigmoid'))
    return model

def get_base_model(shape, output):
    base_model = ResNet152(weights="imagenet", input_shape=shape, include_top=False)
    base_model.trainable = True

    model = tf.keras.Sequential()
    model.add(base_model)
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(output, activation="sigmoid"))
    return model

def get_generators(size, batch_size):
    image_dir = "../MIRFLICKR/mirflickr/"
    csv_dir = "./output.csv"

    df = pd.read_csv(csv_dir)
    X = np.array(df["Image"])
    y = np.array(df.drop("Image", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    train_generator = Generator(X_train, y_train, image_dir, batch_size, size)
    test_generator = Generator(X_test, y_test, image_dir, batch_size, size)
    return train_generator, test_generator

if __name__ == "__main__":
    SIZE = 224
    BATCH_SIZE = 8

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    train_generator, test_generator = get_generators(SIZE, BATCH_SIZE)
    x, y = train_generator[1]
    print(x[0].shape)
    print(y[0].shape[0])

    # model = get_model(x[0].shape, y[0].shape[0])
    model = get_base_model(x[0].shape, y[0].shape[0])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    # history = model.fit(X_train, y_train, epochs=15, callbacks=[es_callback], validation_split=0.3, batch_size=16)
    # history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)
    history = model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=30)

