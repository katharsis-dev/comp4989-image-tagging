from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('marilyn.jpg').astype('int32')
if image.ndim == 3:
    image = image.mean(axis=2)  # convert to grayscale

# horizontal transformation - highlights vertical edges
horizontal_edges = ndimage.sobel(image, 0)

# vertical transformation - highlights horizontal edges
vertical_edges = ndimage.sobel(image, 1)

# magnitude of the gradient for each pixel
magnitude = np.sqrt(horizontal_edges ** 2 + vertical_edges ** 2)
magnitude *= 255.0 / np.max(magnitude)  # normalize the magnitude values to 0-255

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plt.gray()  # show in grayscale
plots = [image, horizontal_edges, vertical_edges, magnitude]
titles = ['Original', 'Horizontal', 'Vertical', 'Sobel Magnitude']
for i in range(4):
    axs[int(i / 2), i % 2].imshow(plots[i])
    axs[int(i / 2), i % 2].set_title(titles[i])

plt.show()
