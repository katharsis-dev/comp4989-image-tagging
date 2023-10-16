import tensorflow
from tensorflow.keras import utils
from tensorflow.keras.layers import Resizing, Rescaling, RandomRotation, RandomCrop
import numpy as np
import pandas as pd

class Generator(tensorflow.keras.utils.Sequence) :

    def __init__(self, x, y, image_dir, batch_size, image_size) :
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.image_size = image_size

    def __len__(self) :
        return int((len(self.x) / self.batch_size))

    def __getitem__(self, index) :
        X_dataset = []
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        for i in range(start, end):
            # /dir/im1.jpg
            img = utils.load_img(self.image_dir + 'im' + str(self.x[i]) + '.jpg', target_size=(self.image_size, self.image_size, 3))
            img = utils.img_to_array(img)
            img = img/255.
            X_dataset.append(img)
        X_dataset = np.array(X_dataset)
        # aug_images = np.array(image_augmentation(X_dataset))
        # print(X_dataset.shape)
        # print(aug_images.shape)
        # print(np.concatenate([X_dataset, aug_images], axis=0).shape)
        # X_dataset = image_augmentation(X_dataset)
        # X_dataset = np.concatenate([X_dataset, aug_images])
        y = self.y[start:end]
        return X_dataset, y

if __name__ == "__main__":
    SIZE = 200
    image_dir = "../MIRFLICKR/mirflickr/"
    csv_dir = "./output.csv"

    df = pd.read_csv(csv_dir)
    X = np.array(df["Image"])
    y = np.array(df.drop("Image", axis=1))
    train_generator = Generator(X, y, image_dir, 16, SIZE)
    train_generator[1]
    # print(train_generator[1])
