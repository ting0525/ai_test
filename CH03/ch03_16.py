import numpy as np
x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
# 將大於 3 的數值印出
print(x[x>3])

x1 = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
y1 = np.array([[1],[3],[6],[9]])
# 判斷 x1 內的元素是否大於 y1
print(x1>y1)
# 將 x1 內的元素大於 y1 的元素印出
print(x1[x1>y1])


