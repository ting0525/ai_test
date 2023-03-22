import numpy as np
a = np.array([0, 30, 45, 60, 90])
print('不同角度的正弦值：')
# 乘 pi/180 轉成弧度
print(np.sin(a * np.pi / 180))
print('不同角度的餘弦值：')
print(np.cos(a * np.pi / 180))
print('不同角度的正切值：')
print(np.tan(a * np.pi / 180))

