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
zero_arr = np.zeros((28,28)) # the numpy array of zero values with the shape (28,28)

# Creating ones array
ones_arr = np.ones((28,28)) # the numpy array of one values with the shape (28,28)

# Creating noisy image
noisy_arr = np.random.rand(64, 64, 3) # the numpy array of random values from 0 to 1
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

As common in Python, indexing starts from 0. To access an element in a 2D array, you should put two numbers in a square brackets. The first you accessing the rows, then the columns. You can also access only rows by one index and only columns by using [:, n], where *n* - the index of the column.

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

Similarly to Indexing, we can extract sections of an array using slicing operations. Specifying the start index and the end index of a row or a column, we will select everything between them. Using the ":" index we will select the whole column or row.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
                
# Slicing the last two columns
print(arr[:,1:3]
# Output:
# [[2 3]
#  [5 6]
#  [8 9]]

# Slicing a section of the array
print(arr[1:3, 1:3])
# Output:
# [[5 6]
#  [8 9]]
```


## Shapes and Dimensions

The shape of a NumPy array is a tuple of integers that gives the size of the array along each dimension. For example, a 2-dimensional array with 3 rows and 4 columns would have a shape of (3, 4). A 1-dimensional array with 5 elements would have a shape of (5,). You can get the shape of a NumPy array using the shape attribute.

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr) # Output: [1 2 3 4 5]

# Understanding the shape
print(arr.shape) # Output: (5,)

# Reshaping an array
arr = arr.reshape(5, 1)
print(arr)
# Output:
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]]
```

## Operations on Arrays

Something Something

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
                
# The sum of all elements in the array
sum_value = arr.sum() # or np.sum(arr)
print(sum_value) # Output: 45

# The sum of the column elements
sum_values = arr.sum(axis=0) # or np.sum(arr, axis=0)
print(sum_values) # Output: [12 15 18]

# The sum of the row elements
sum_values = arr.sum(axis=1) # or np.sum(arr, axis=1)
print(sum_values) # Output: [6 15 24]

# Multiplication of arrays
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
C = A * B
print(C) # Output: [4 10 18]

# Division of arrays
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
C = A / B
print(C) # Output: [0.25 0.4 0.5]
```

Something Something: Three important functinos: np.rot90, np.concatenate, np.where

np.rot90 - When we have an image as an array, we can rotate the whole image by 90 degrees

We use it with two arguments. The array to rotate and the number of times to rotate the array by 90 degrees.

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])


b = np.rot90(a)
print(b) # Output: [[3 6 9]
                    [2 5 8]
                    [1 4 7]]


np.concatenate - When we have several images what we want to concatenate together in order to get a new image, consisting of them together

We will use a simple example of two small arrays

```python
a = np.array([[0,0,0],
              [0,0,0]])
b = np.array([[1,1,1],
              [1,1,1]])
c = np.concatenate([a,b], axis=0)
print(c) # Output: [[0 0 0]
                   [0 0 0]
                   [1 1 1]
                   [1 1 1]]
```


### Contributing

We welcome contributions to this repository. If you have suggestions for new examples or tutorials, or if you have found a bug, please open
