from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import utils
from generator import Generator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    SIZE = 300
    image_dir = "../MIRFLICKR/mirflickr/"
    csv_dir = "./output.csv"

    df = pd.read_csv(csv_dir)
    X = np.array(df["Image"])
    y = np.array(df.drop("Image", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    train_generator = Generator(X_train, y_train, image_dir, 16, SIZE)
    print("Train Generator", train_generator)
    test_generator = Generator(X_test, y_test, image_dir, 15, SIZE)

    model = Sequential()
    # crystal clear to explain what Convolutional Layer is doing, what Pooling Layer is doing, and what
    # connected layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # the 25 label categories for movies
    # 24 labels for mirflickr
    # for activiation, do not use softmax for multi-class problems
    model.add(Dense(24, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=30)

    # display_training(history)
    # test_model_with_image(model, df)

    model.save("oct5_6pm_model1.h5")

    _, acc = model.evaluate(X_test, y_test)
    print(f"Accuracy: {acc * 100.00}%")

    # save model

# def test_model_with_image(model, df):
#     SIZE = 200
#     img = utils.load_img('test_img.jpg', target_size=(SIZE, SIZE, 3))
#     img = utils.img_to_array(img)
#     img = img/255.
#     plt.imshow(img)
#     img = np.expand_dims(img, axis=0)
#
#     classes = np.array(df.columns[2:])
#     prob = model.predict(img)
#     sorted_cats = np.argsort(prob[0])[:-11:-1]
#
#     for i in range(10):
#         print(f"{classes[sorted_cats[i]]} - {prob[0][sorted_cats[i]]}")


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
