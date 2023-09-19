## Numpy Notes from GPT
<h3><strong>Installing NumPy:</strong></h3>
If you haven't already installed NumPy, you can do so using pip: 

#### `pip install numpy`

<h3><strong>Importing NumPy:</strong></h3>
import to environment being used 'import numpy as np'
'np' is the alias of numpy that is commonly used convention

<h3><strong>Creating NumPy Arrays:</strong></h3>
An Example of making a vector (1D Array) and of Matrix (2D Array)

```
#Creating a 1D array (vector)
arr1 = np.array([1, 2, 3, 4, 5])

#Creating a 2D array (matrix)
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```

<h3><strong>Arrays Attributes:</strong></h3>

```
print(arr1.shape)  # Shape of arr1
print(arr2.shape)  # Shape of arr2
print(arr2.ndim)   # Number of dimensions
print(arr2.dtype)  # Data type of elements
```

Numpy has different attributes that are able to give more information abuot their shape, size and data type.

<h3><strong>Array Operations:</strong></h3>

You can make it perform + - / and more
```
# Element-wise addition
result = arr1 + arr1
# Element-wise multiplication
result = arr1 * 2
```
<h3><strong>Array Indexing and Slicing:</strong></h3>
Can use the arrays to index and slice (access the elements and subarrays)

```
element = arr1[2]        # Access the element at index 2
subarray = arr2[:2, 1:]  # Get a subarray of arr2
```

<h3><strong>Array Functions:</strong></h3>
Numpy has many mathematical functions that work on arrays

```
mean_value = np.mean(arr1)  # Calculate the mean of arr1
max_value = np.max(arr2)    # Find the maximum value in arr2
```

<h3><strong>Array Manipulation:</strong></h3>
Numpy has functions with capabilites to reshape, transpose, and concatenate arrays

```
reshaped = arr1.reshape(5, 1)  # Reshape arr1 into a 2D array
transposed = arr2.T            # Transpose arr2
concatenated = np.concatenate((arr1, arr1))  # Concatenate arrays
```

<h3><strong>Random Number Generation:</strong></h3>
Numpy has functions that are able to generate random numbers and arrays

`random_array = np.random.rand(3, 3)  # Create a 3x3 array of random values`

<h3><strong>Broadcasting:</strong></h3>
Numpy lets you perform operations between arrays of different shapes, automatically aligning them when possible

```
a = np.array([1, 2, 3])
b = 2
```
