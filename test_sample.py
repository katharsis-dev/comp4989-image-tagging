import pandas as pd
import numpy as np
from tensorflow.keras import utils
import tensorflow as tf



def get_image_and_label_mirflickr(df, entry):
    image_dir = "../MIRFLICKR/mirflickr/"
    prefix = "im"
    column = "Image"
    image_size = 224
    img = utils.load_img(image_dir + prefix + str(df.iloc[entry][column]) + '.jpg', target_size=(image_size, image_size, 3))
    img = utils.img_to_array(img)
    img = img/255.
    return np.array([img])


def predict(model, image):
    print(model.predict(image))


if __name__ == "__main__":
    df = pd.read_csv("../MIRFLICKR/output.csv")
    entry = 1
    model_path = "./models/resnet50_mirflickr_1.keras"

    image = get_image_and_label_mirflickr(df, entry)
    print(image.shape)

    model = tf.keras.models.load_model(model_path)

    print("Labels")
    print(df.iloc[entry])
    prediction = predict(model, image)



