import matplotlib.pyplot as plt
import numpy as np
# 從x=0到10當中產生50個x值
x = np.linspace(0,10,50)
y = np.sin(x)
plt.scatter(x, y, c='blue', marker = '*')
plt.show()

