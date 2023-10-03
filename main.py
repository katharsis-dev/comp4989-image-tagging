import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from sklearn.metrics import classification_report

api = KaggleApi()
api.authenticate()

api.dataset_download_files('andrewmvd/animal-faces', path='data', unzip=True)

train_ds = keras.utils.image_dataset_from_directory(
    'data/afhq/train',
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset='training',
)

test_ds = keras.utils.image_dataset_from_directory(
    'data/afhq/val',
    seed=123,
    image_size=(128, 128),
    batch_size=1,
    shuffle=False
)

model = Sequential()
model.add(layers.Rescaling(1./255))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=test_ds)
model.evaluate(train_ds)
model.evaluate(test_ds)

pred = model.predict(test_ds).argmax(axis=1)

print(classification_report(pred, test_ds.classes))
