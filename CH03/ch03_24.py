import numpy as np

a = np.array([[3, 7], [9, 1]])
print('欲排列的陣列為：')
print(a)
print('使用sort()排列後陣列為：')
print(np.sort(a))
print('按照指定軸=0(按行)排序：')
print(np.sort(a, axis=0))
print('按照指定軸=1(按列)排序：')
print(np.sort(a, axis=1))
# 在 sort 函數中指定排序內容
dt = np.dtype([('name','S10'), ('age', int)])
a = np.array([("Jacky", 21),("Peter", 25),("Many", 17),
              ("Andy", 27)],dtype=dt)
print('原來的陣列順序內容：')
print(a)
print('按 name 排序後的陣列內容：')
print(np.sort(a, order='name'))

