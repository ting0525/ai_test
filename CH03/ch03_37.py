import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,50)
y1 = np.sin(x)
y2 = np.cos(x)
# marker 的顏色設成半透明
plt.scatter(x, y1, c='blue', marker = '*', alpha = 0.5)
# marker 有加外框
plt.scatter(x, y2, c='red',edgecolor='green', linewidths = 2 ,marker = 'o')
plt.show()

