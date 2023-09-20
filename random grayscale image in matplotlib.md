# Creating a random grayscale image and show it in matplotlib

### Import the necessary libraries:
```
import numpy as np
import matplotlib.pyplot as plt
```

### Create a random grayscale image:
```
# Define the dimensions of the image (e.g., 100x100 pixels)
width, height = 100, 100

# Generate random grayscale values between 0 (black) and 255 (white)
random_grayscale_image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
```

### Displaying the image using Matplotlib
# Create a Matplotlib figure and axis
plt.figure(figsize=(6, 6))
plt.imshow(random_grayscale_image, cmap='gray', vmin=0, vmax=255)
```
# Add a title
plt.title('Random Grayscale Image')

# Show the image
plt.axis('off')  # Turn off the axis labels
plt.show()
```

### Things to Note
We first import NumPy for generating random data, Matplotlib for displaying the image, and create a random grayscale image using np.random.randint().

We specify the dimensions of the image (e.g., 100x100 pixels) and use np.random.randint() to generate random grayscale values between 0 and 255, where 0 represents black and 255 represents white.

We create a Matplotlib figure and axis, display the image using plt.imshow(), and specify the color map (cmap='gray') to display it as grayscale. The vmin and vmax arguments are set to 0 and 255 to ensure the full range of grayscale values is used.

We add a title to the plot using plt.title() and turn off the axis labels using plt.axis('off') to make it look like a pure image.

When you run this code, it will generate a random grayscale image and display it using Matplotlib.

