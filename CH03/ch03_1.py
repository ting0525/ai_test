import numpy as np

arr1 = np.array([0,1,2]) # 傳入 tuple 創建一維陣列
print(type(arr1))        # 列出 arr1 類別型態
print(arr1.shape)        # 列出 arr1 陣列大小
print(arr1.ndim)         # 列出 arr1 維度大小
print(arr1)              # 列出 arr1 內容
print(arr1[1])           # 找出arr1索引值為 1 的內容

arr2 = np.array([[0,1,2],
                 [4,5,6]]) # 傳入 tuple 創建二維陣列
print(type(arr2))          # 列出 arr2 類別型態
print(arr2.shape)          # 列出 arr2 陣列大小
print(arr2.ndim)           # 列出 arr2 維度大小
print(arr2)                # 列出 arr2 內容
print(arr2[1][1])          # 找出 arr2索引值為 1 的內容


