import numpy as np

A = np.array([
    [1,2],
    [3,4]])
B = np.array([
    [5,6],
    [7,8]])
# 水平合併
print(np.hstack((A,B)))
# 垂直合併
print(np.vstack((A,B)))

