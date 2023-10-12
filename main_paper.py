import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet import ResNet152, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, Activation

from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

data = pd.read_csv("../Movies-Poster_Dataset/train.csv")
# data = data.head(100)
print(data.shape)

image_height = 300
image_width = 300


def get_data():
    store_list = []
    for i in tqdm(range(data.shape[0])):
        path = "../Movies-Poster_Dataset/Images/" + data["Id"][i] + ".jpg"
        image_check = image.load_img(path, target_size=(image_height, image_width))
        image_check = image.img_to_array(image_check)

        # scaling the images
        image_check = image_check/255
        store_list.append(image_check)

    x = np.array(store_list)
    print(x.shape)

    # now setup the y
    y = data.drop(columns = ["Id", "Genre"])
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)

    return X_train, X_test, y_train, y_test

def get_base_model(shape, output):
    base_model = ResNet50(weights="imagenet", input_shape=shape, include_top=False)
    base_model.trainable = False

    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(output, activation="sigmoid"))
    return model


def get_model(shape, output):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(96, 11, strides=4, padding='same', input_shape=shape))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    # Create a Sequential model

    # # First Convolutional Layer
    # model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=shape, padding='valid', activation='relu'))
    #
    # # Second Convolutional Layer
    # model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    #
    # # Third Convolutional Layer
    # model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    #
    # # Fourth Convolutional Layer
    # model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    #
    # # Fifth Convolutional Layer
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    #
    # # Flatten the output for fully connected layers
    # model.add(Flatten())
    #
    # # Fully Connected Layer 1
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # # Fully Connected Layer 2
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))

    # Output Layer (assuming a specific number of classes)
    model.add(Dense(output, activation='sigmoid'))
    return model

if __name__ == "__main__":
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    X_train, X_test, y_train, y_test = get_data()
    model = get_base_model(X_train[0].shape, 25)
    #
    # model = get_model(X_train[0].shape, 25)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    # es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    # # history = model.fit(X_train, y_train, epochs=15, callbacks=[es_callback], validation_split=0.3, batch_size=16)
    history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)
