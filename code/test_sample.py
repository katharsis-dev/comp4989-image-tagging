import variables
import pandas as pd
import numpy as np
from tensorflow.keras import utils
import tensorflow as tf



def get_image_and_label_mirflickr(df, entry):
    image_dir = variables.MIR_FLICKR_DIR
    prefix = "im"
    column = "Image"
    image_size = 224
    img = utils.load_img(image_dir + prefix + str(df.iloc[entry][column]) + '.jpg', target_size=(image_size, image_size, 3))
    img = utils.img_to_array(img)
    img = img/255.
    return np.array([img])


if __name__ == "__main__":
    df = pd.read_csv(variables.MIR_FLICKR_CSV)
    entry = 1
    model_path = variables.MODEL_LOAD

    image = get_image_and_label_mirflickr(df, entry)
    print("Image Shape:", image.shape)

    model = tf.keras.models.load_model(model_path)

    if model:
        result_df = df.drop("Image", axis=1).iloc[[entry]].reset_index(drop=True)

        prediction = np.around(model.predict(image), decimals=5)

        result_df.loc[len(result_df)] = prediction.flatten()
        print(f"Labels for Image {entry}")
        print(result_df)
        print("Row 0: Actual labels provided from the dataset")
        print("Row 1: Predicted values from the loaded model")



