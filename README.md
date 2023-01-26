Welcome to the Numpy: Indexing, Slicing, and Arrays repository! Here you will find tutorials and code examples that will help you master the powerful array manipulation capabilities of the Numpy library.

Indexing

Learn how to select specific elements from an array using indexing and slicing operations. We will cover how to access individual elements, rows, and columns, as well as how to use boolean indexing to select elements that match certain conditions.

Copy code
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Accessing an element
print(arr[1, 2]) # Output: 6

# Accessing a row
print(arr[1]) # Output: [4, 5, 6]

# Accessing a column
print(arr[:, 1]) # Output: [2, 5, 8]

# Boolean indexing
print(arr[arr > 5]) # Output: [6, 7, 8, 9]
Slicing

Discover how to extract sections of an array using slicing operations. We will cover how to slice arrays along different axes and how to use the step parameter to select every n-th element.

Copy code
import numpy as np

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
Arrays

Learn how to create arrays, understand the difference between 1D, 2D and 3D arrays, learn how to inspect the shape and size of an array, understand the difference between reshaping and resizing arrays, and learn how to concatenate and split arrays

Copy code
import numpy as np

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
Requirements

This repository requires Numpy library to be installed, the examples and tutorials are written in Python. Make sure you have the latest version of Numpy before running the examples.

Contributing

We welcome contributions to this repository. If you have suggestions for new examples or tutorials, or if you have found a bug, please open
