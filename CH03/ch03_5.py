import numpy as np
# 默認底數是 10
arr1 = np.logspace(1.0, 2.0,num=10)
print (arr1)
# 底數為 2
arr2 = np.logspace(1,5,num=5,base=3)
print (arr2)

