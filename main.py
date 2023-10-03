import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

data = pd.read_csv("../Movies-Poster_Dataset/train.csv")
data = data.head(100)
print(data.shape)

image_height = 350
image_width = 350


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


def get_model(shape):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(25, activation="sigmoid"))

    
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    model = get_model(X_train[0].shape)

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es_callback], validation_split=0.3)

