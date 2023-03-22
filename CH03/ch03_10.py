import numpy as np

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(arr)
# 從維度 1 開始截取
print(arr[1:])
print(arr[...,1])   # 第2行元素
print(arr[1,...])   # 第2列元素
print(arr[...,1:])  # 第2行之後的所有元素


