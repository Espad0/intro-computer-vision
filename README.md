**Intro №1: Numpy**

This is the first part of the tutorial series "Intro To Computer Vision". Here, you will find the main theory along with code examples that will help you master the basics of Numpy: Arrays, Mathematical Operations, Indexing, Other Ops.

Let's get started!

```python
import numpy as np
```

## Lists vs Arrays

In Python, we can create lists and perform some mathematical operations on them. However, it doesn't always give us what we want

```python
# Lists
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]

# Lists addition
sum_list = list1 + list2
print(sum_list) # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Lists multiplication
mult_list = list1 * 3
print(mult_list) # Output: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
```

Numpy's arrays allow us to perform these operations much easier

```python
# Arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])

# Arrays addition
sum_arr = arr1 + arr2
print(sum_arr) # Output: [7, 9, 11, 13, 15]

# Arrays multiplication
mult_arr = arr1 * 3
print(mult_arr) # Output: [3, 6, 9, 12, 15]
```


## Creating Arrays

In Computer Vision we usually work with images. There are ways to create simple images using Numpy.

```python
# Creating zero array
zero_arr = np.zeros((28,28)) # This will give us the numpy array of zero values with the shape (28,28)

# Creating ones array
ones_arr = np.ones((28,28)) # This will give us the numpy array of one values with the shape (28,28)

# Creating noisy image
noisy_arr = np.random.rand(64, 64, 3) # This will give us the numpy array of random values from 0 to 1
```

Sometimes the arrays come from your data. RGB images are such data: 3-channel arrays.
For example, let us download a piece of data from the famous MNIST dataset.

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
data_sample = x_train[0] # Take the first element of the data
print(data_sample) # Outputs an array of 28x28

[[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,  13,  14,  34,  98, 119,  47, 116, 191, 163,  56,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   2,  29,  55, 118, 152, 227, 233, 236, 243, 230, 175, 238, 239, 167,  40,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,  62, 240, 253, 253, 253, 253, 248, 248, 251, 154,  78,  75,  50,  27,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,  28, 195, 224, 238, 253, 242, 150, 151, 206,  91,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   4,  59,  83, 150, 243, 165,   6,   6,  52,  37,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   1,   4,  85, 226, 157,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,  25, 165, 235,  48,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  36, 226, 206, 135,  58,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68, 225, 249, 202,  75,  10,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  43, 161, 230, 208,  94,   5,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  21, 136, 252, 223,  42,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  31,  84, 167, 252, 231,  44,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  22, 104, 187, 226, 243, 252, 204,  16,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   9,  70, 180, 241, 251, 253, 233, 139,  34,   1,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   1,  16,  50, 138, 238, 253, 248, 234, 155,  44,   1,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   8,  36,  68, 195, 234, 253, 252, 230, 155,  56,   5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,  52, 176, 234, 251, 228, 203, 171,  95,  12,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,  33,  93, 105, 104,  80,  56,  27,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]
 ```



## Indexing

In case you need to select specific elements from an array to perform some operations on them - you will use indexing and slicing operations. 
In Numpy you can access individual elements, rows, and columns, as well as using boolean indexing to select elements that match certain conditions.

As common in Python, indexing starts from 0. To access an element, you should put two numbers in a square brackets. The first you accessing the rows, then the columns. You can also access only rows by one index and only columns by using [:, n], where *n* - the index of the column.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Accessing the second row and the third column
print(arr[1, 2]) # Output: 6

# Accessing the third row
print(arr[2]) # Output: [7, 8, 9]

# Accessing the column
print(arr[:, 1]) # Output: [2, 5, 8]

# Boolean indexing
print(arr[arr > 5]) # Output: [6, 7, 8, 9]
```


## Slicing

Discover how to extract sections of an array using slicing operations. We will cover how to slice arrays along different axes and how to use the step parameter to select every n-th element.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Slicing a section of the array
print(arr[1:3, 1:3])
# Output:
# [[5 6]
#  [8 9]]

# Slicing with step parameter
print(arr[::2, ::2])
# Output:
# [[1 3]
#  [7 9]]
```


## Arrays

Learn how to create arrays, understand the difference between 1D, 2D and 3D arrays, learn how to inspect the shape and size of an array, understand the difference between reshaping and resizing arrays, and learn how to concatenate and split arrays

```python
# Creating an array
arr = np.array([1, 2, 3, 4, 5])
print(arr) # Output: [1 2 3 4 5]

# Understanding the shape and size of an array
print(arr.shape) # Output: (5,)
print(arr.size) # Output: 5

# Reshaping an array
arr = arr.reshape(5, 1)
print(arr)
# Output:
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]]

# Concatenating arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.concatenate([arr1, arr2])
print(arr3) # Output: [1 2 3 4 5 6]
```

### Requirements

This repository requires Numpy library to be installed, the examples and tutorials are written in Python. Make sure you have the latest version of Numpy before running the examples.

### Contributing

We welcome contributions to this repository. If you have suggestions for new examples or tutorials, or if you have found a bug, please open
