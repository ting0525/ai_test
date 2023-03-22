import numpy as np

# 從 1-10產生10個值
arr1 = np.linspace(1,10,10)
print(arr1)
# 從 1-10產生5個值
arr2 = np.linspace(1,10,5)
print(arr2)
# 從 1-1產生10個值
arr3 = np.linspace(1,1,10)
print(arr3)
# 從 1-10產生5個值,不包含10
arr4 = np.linspace(1, 10,  5, endpoint =  False)
print(arr4)

