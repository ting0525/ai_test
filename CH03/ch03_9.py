import numpy as np
arr = np.arange(10)
# 設定切片範圍
s1 = slice(0,8,2)
print(arr[s1])
# 意思與上面同
s2 = arr[0:8:2]
print(s2)
# 從索引2開始往後提取
s3 = arr[2:]
print(s3)


