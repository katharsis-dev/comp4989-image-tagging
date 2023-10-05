import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import BatchNormalization

from keras.models import load_model
from keras.models import model_from_json, model_from_yaml

def display_training(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label="Training Loss")
    plt.plot(epochs, val_loss, 'r', label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # plt.plot(epochs, acc, 'y', label="Training Acc")
    # plt.plot(epochs, val_acc, 'r', label="Validation Acc")
    # plt.title("Training and Validation Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

def train_and_save_model():
    image_dir = "dataset/images/"
    csv_dir = "dataset/train.csv"
    # read .csv to get data regarding each image in our dataset
    df = pd.read_csv(csv_dir)
    print(df.head())
    print(df.columns)

    # load limited set of datapoints first to prevent memory issues
    # df = df.iloc[:5000] 

    SIZE = 200
    X_dataset = []
    for i in tqdm(range(df.shape[0])):
        img = image.load_img(image_dir + df['Id'][i] + '.jpg', target_size=(SIZE, SIZE, 3))
        img = image.img_to_array(img)
        img = img/255.
        X_dataset.append(img)

    # create an array of size (5000, 200, 200, 3) - 2
    X = np.array(X_dataset)

    # drop ID and Genre as we don't care about them
    # dataset is 

    y = np.array(df.drop(['Id', 'Genre'], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3)

    model = Sequential()
    # crystal clear to explain what Convolutional Layer is doing, what Pooling Layer is doing, and what
    # connected layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(SIZE, SIZE, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # the 25 label categories
    # for activiation, do not use softmax for multi-class problems
    model.add(Dense(25, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
    # display_training(history)
    test_model_with_image(model, df)

    _, acc = model.evaluate(X_test, y_test)
    print(f"Accuracy: {acc * 100.00}%")

    # save model
    model.save("full_model.h5")

def test_model_with_image(model, df):
    SIZE = 200
    img = image.load_img('test.jpg', target_size=(SIZE, SIZE, 3))
    img = image.img_to_array(img)
    img = img/255.
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)

    classes = np.array(df.columns[2:])
    prob = model.predict(img)
    sorted_cats = np.argsort(prob[0])[:-11:-1]

    for i in range(10):
        print(f"{classes[sorted_cats[i]]} - {prob[0][sorted_cats[i]]}")


def main():
    train_and_save_model()


if __name__ == "__main__":
    main()


# talk about preprocessing
# talk about model trianing -> CNN
    # discuss what a CNN is
    # what is the feature detector doing?
    # why CNN? Why not just use a neural network?
    # how many layers for CNN
    # talk about the hidden layer
    #
# Scope issue ->

# discuss what CNN is first
# if time, then you can discuss the 2017 -> 2024 changes relating to CNN/resnet/etc
