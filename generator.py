from sklearn.model_selection import train_test_split
import keras
import keras.utils as image
import pandas as pd
import numpy as np

class Generator(keras.utils.Sequence):

    def __init__(self, x_df, y_df, image_dir, batch_size, size=200) -> None:
        super().__init__()
        self.x_df = x_df
        self.y_df = y_df
        self.size = size
        self.batch_size = batch_size
        self.image_dir = image_dir

    def __len__(self):
        return int((len(self.x_df)/self.batch_size))

    def __getitem__(self, index):
        X_dataset = []
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        for i in range(start, end):
            # /dir/im1.jpg
            img = image.load_img(self.image_dir + 'im' + str(self.x_df[i]) + '.jpg', target_size=(self.size, self.size, 3))
            img = image.img_to_array(img)
            img = img/255.
            X_dataset.append(img)

        # create an array of size (~25000, 200, 200, 3) - 2
        X = np.array(X_dataset)

        # drop ID and Genre for movies as we don't care about them
        # For mirflickr, drop Image
        y = np.array(self.y_df)[start:end]
        return X, y


if __name__ == "__main__":
    SIZE = 200
    image_dir = "../MIRFLICKR/mirflickr/"
    csv_dir = "./output.csv"

    df = pd.read_csv(csv_dir)
    X = np.array(df["Image"])
    y = np.array(df.drop("Image", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    train_generator = Generator(X_train, y_train, image_dir, 16, SIZE)
    test_generator = Generator(X_test, y_test, image_dir, 15, SIZE)

    print(len(train_generator))
    x, y = train_generator[1]
    print(x.shape)
    print(y.shape)
