import tensorflow as tf
import matplotlib.pyplot as plt

img = tf.keras.utils.load_img("test_img.jpg", target_size=(5, 5, 3))
img = tf.keras.utils.img_to_array(img)




print(img)
resized_image = tf.keras.preprocessing.image.array_to_img(img)

# Display the resized image
plt.plot(1, 2, 2)
plt.title('Resized Image')
plt.imshow(resized_image)
plt.axis('off')

plt.show()
