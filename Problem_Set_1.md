
To find code [Click Here for Problem 1](https://github.com/StoicSophist/Math-Data-Science/blob/main/Problem_Set_1.ipynb)
---
title: MNIST Data Processing
date: 2023-10-17
author: Sofia Vanegas
---

# Introduction

This Google Colab aims to demonstrate data processing for the MNIST dataset. It uses Python with PyTorch and GPU acceleration. The notebook showcases loading the MNIST dataset, data preprocessing, and some initial experiments.

## Colab Link

You can view this notebook on Google Colab by following this link: [Open In Colab](https://colab.research.google.com/github/StoicSophist/Math-Data-Science/blob/main/Problem_Set_1.ipynb)

# Load MNIST

In this section, we'll load the MNIST dataset, perform data preprocessing, and visualize the data.

## Part 1.a Load MNIST and show montage

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
from skimage.io import imread
```
# Define GPU functions and plot functions
```
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))

def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))

def plot(x):
    # Function to plot an image
    # ...

def montage_plot(x):
    # Function to plot a montage of images
    # ...
```

# Load MNIST dataset
```
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)
```

# Perform data preprocessing
```
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:, None, :, :] / 255
X_test = X_test[:, None, :, :] / 255
```

# Reshape data
```
X = X.reshape(X.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
```

# Transfer data to GPU
```
X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)
```


# Introduction

The Jupyter Notebook appears to be aimed at processing the MNIST dataset using Python with PyTorch, focusing on GPU acceleration. It covers several steps, including loading the MNIST dataset, data preprocessing, matrix operations, and an optimization experiment using gradient descent.

## Load MNIST

In the "Load MNIST" section, the code does the following:

1. It imports necessary libraries, including NumPy, Matplotlib, and PyTorch, and specific modules from the torchvision and skimage libraries.

2. It defines two functions: `GPU(data)` and `GPU_data(data)`. These functions transfer data to the GPU and specify whether or not gradient calculations are required for that data.

3. The `plot(x)` and `montage_plot(x)` functions are defined to visualize images. `plot(x)` displays a single image, and `montage_plot(x)` creates a montage of images.

4. The MNIST dataset is loaded using the `datasets.MNIST` function from torchvision. This function downloads the MNIST dataset and stores it in a "train_set" and "test_set."

5. Data preprocessing is performed on the images to normalize pixel values to the range [0, 1]. The images are also reshaped from 28x28 to a flat vector of size 784.

6. The data (X) and labels (Y) are transferred to the GPU for processing.

## Experiment 1: Matrix Operations

In the "Experiment 1: Matrix Operations" section, the code performs the following steps:

1. It defines a batch size of 64.

2. Matrix operations are performed on the data. It first extracts a single column of data (x) and computes its shape. Then, it generates a random matrix (M) with dimensions (10, 784) and performs matrix multiplication (M @ x).

3. The code updates the data (x) by slicing it based on the batch size. Another random matrix (M) is generated, and matrix multiplication (M @ x) is performed.

4. The code calculates the element-wise maximum value along the first dimension of the resulting matrix (y) using `torch.argmax(y, 0)`.

5. Finally, it calculates the accuracy by comparing the predicted values (y) with the true labels (Y) for the specified batch size.

## Experiment 2: Gradient Descent

In the "Experiment 2: Gradient Descent" section, the code implements a gradient descent experiment to optimize a matrix for improved accuracy. It does the following:

1. Initializes variables `m_best` and `acc_best` to keep track of the best-performing matrix and its associated accuracy.

2. Enters a loop that runs 100,000 iterations.

3. In each iteration, a small random matrix `m_random` is generated. The code then updates the matrix `m` by taking a step (`step`) in the direction of this random matrix.

4. It computes the predictions (y) by multiplying the updated matrix `m` with the entire dataset `X`.

5. The code calculates the element-wise maximum value along the first dimension of the predictions (y) and stores it in `y`.

6. The accuracy is calculated by comparing the predicted values (y) with the true labels (Y).

7. If the accuracy is greater than the best accuracy observed so far (`acc_best`), the code updates `m_best` and `acc_best`.

Overall, this code demonstrates loading, preprocessing, and performing matrix operations on the MNIST dataset. It also illustrates a simple optimization experiment using gradient descent to find the best matrix for classification.
