import numpy as np

arr1 = np.array([0,1,2,3,4,5,6,7,
                 8,9,10,11,12,13,14,15])
print(arr1.shape)
arr2 = arr1.reshape(2,8)  # 將維度改為 2x8
print(arr2)
print(arr2.shape)
arr3 = arr1.reshape(4,4)  # 將維度改為 4x4
print(arr3)
print(arr3.shape)

