Welcome to the **Intro №1: Numpy**! Here you will find tutorials and code examples that will help you master the powerful array manipulation capabilities of the Numpy library.

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

Numpy's arrays allows us to perform these operations much easier

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

Usually, the arrays come from your data. RGB images are such data: 3-channel arrays.
For example, let us download a piece of data from the famous dataset MNIST.

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
data_sample = x_train[0] # Take the first element of the data
print(data_sample) # Outputs an array of 28x28

[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255 247 127   0   0   0   0]
 [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0]
 [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82  82  56  39   0   0   0   0   0]
 [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253 253 207   2   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201  78   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]
 
```



## Indexing

Learn how to select specific elements from an array using indexing and slicing operations. We will cover how to access individual elements, rows, and columns, as well as how to use boolean indexing to select elements that match certain conditions.

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Accessing an element
print(arr[1, 2]) # Output: 6

# Accessing a row
print(arr[1]) # Output: [4, 5, 6]

# Accessing a column
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
