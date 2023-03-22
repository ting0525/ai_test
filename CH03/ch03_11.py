import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
arr1 = arr.reshape((3,3))
print(arr1.shape)
print(arr1)
# 取第一維的索引1到索引2之間的元素，也就是第二行
# 取第二維的索引1到索引3之間的元素，也就是第二列和第三列
arr2 = arr1[1:2, 1:3]
print(arr2)

