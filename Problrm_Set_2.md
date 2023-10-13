To view code [Click Here For Problem Set 2 Code](https://github.com/StoicSophist/Math-Data-Science/blob/main/Problem_Set_2.ipynb)
# Image Processing Report

In this report, we will go through a Python script that loads an image from a URL, performs various image processing tasks, and displays the results using the `imageio`, `matplotlib`, and `scipy.signal` libraries.

## Loading an RGB Image from URL

The script starts by loading an RGB image from a specified URL using the `imageio` library. The image is then displayed using a custom function `plot(x)`:

```python
import numpy as np
import imageio
import matplotlib.pyplot as plt
import scipy.signal

def plot(x):
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray')
    ax.axis('off')
    fig.set size_inches(5, 5)
    plt.show()

image = io.imread("https://ichef.bbci.co.uk/news/976/cpsprodpb/BE83/production/_104617784_division-bell-metal-storm-thorgerson.jpg")

image = image[:, :, :]

plot(image)
```

This code section loads the image and displays it in grayscale. 

## Show Grayscale Copy, Resize Image, and Apply 10 Random Filters with Convolve

After displaying the original image, the script proceeds with image processing. The image is converted to grayscale, and then 10 random filters are applied to it using convolution:

```python
# Convert the image to grayscale
grayscale_image = np.mean(image, axis=2)

plot(grayscale_image)

# Convolve with 10 random filters and show filters and feature maps
num_filters = 10
filter_size = 30

plt.figure(figsize=(12, 6))

for i in range(num_filters):
    random_filter = np.random.randn(filter_size, filter_size)
    filtered_image = scipy.signal.convolve2d(grayscale_image, random_filter, mode='same', boundary='wrap')

    plt.subplot(2, num_filters, i + 1)
    plt.imshow(random_filter, cmap='gray')
    plt.title(f"Filter {i + 1}")

    plt.subplot(2, num_filters, i + num_filters + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f"Feature Map {i + 1}")

plt.show()
```

This section first converts the image to grayscale and displays it. Then, it applies 10 random filters using convolution and displays both the filters and their corresponding feature maps.

This concludes the report on the image processing script. The code showcases how to load and manipulate images, apply filters, and visualize the results using Python and various libraries.
