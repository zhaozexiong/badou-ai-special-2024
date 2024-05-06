import numpy as np

arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 1, 1]])

mean = np.array([np.mean(attr) for attr in arr1.T])
print(mean)
arr1_mean = np.mean(arr1, axis=0)

print(arr1_mean)

arr2 = arr1 - arr1_mean

print(arr2)

print(arr2.size)

print(np.shape(arr2)[1])

print('-' * 8)
a, b = np.linalg.eig(arr1)
print(a)
print(b)
y = np.argsort(a * -1)
print(y)

print('.' * 8)
x = [b[:, y[i]] for i in range(2)]
print(x)

print(np.transpose(x))
