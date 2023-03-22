import numpy as np

a = np.arange(9, dtype=np.float_).reshape(3, 3)
print('第一個陣列值组：')
print(a)
print('第二個陣列值组：')
b = np.array([10, 10, 10])
print(b)
print('兩陣列相加：')
print(np.add(a, b))
print('兩陣列相減：')
print(np.subtract(a, b))
print('兩陣列相乘：')
print(np.multiply(a, b))
print('兩陣列相除：')
print(np.divide(a, b))

