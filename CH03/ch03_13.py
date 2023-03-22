import numpy as np
a = np.arange(0,60,5)
a1 = a.reshape(3,4)
# 列出陣列內部元素
print(a1)
print('拜訪陣列內部元素：')
for x in np.nditer(a):
    print(x)

