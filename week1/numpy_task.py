import numpy as np

# 1 Create a 5x5 NumPy array filled with normally distributed (i.e. µ = 0, σ = 1)
arr = np.random.normal(loc=0, scale=1, size=(5, 5))

# 2 If the value of an entry is greater than 0.09, replace it with its square. Else, replace it with 42
arr[arr > 0.09] = arr[arr > 0.09] ** 2
arr[arr <= 0.09] = 42

print(arr)

# Use slicing to print just the fourth column of your array.
print(arr[:, 3]) 