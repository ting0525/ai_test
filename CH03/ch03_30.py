import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(0.0,2*np.pi)
y1 = np.sin(x1)
x2 = np.linspace(0.0,2*np.pi)
y2 = np.cos(x2)
x3 = np.linspace(0.0,2*np.pi)
y3 = np.sinc(x3)
line1, = plt.plot(x1, y1, 'r')
line2, = plt.plot(x2, y2, 'b-o')
plt.ylim(-2, 2)   # 設定 y 軸範圍
l1 = plt.legend(handles=[line1,line2],labels=['sin','cos'],loc='upper right')
line3, = plt.plot(x3, y3, 'g-x')
# 此行加入會移走 l1
plt.legend(handles=[line3], labels=['tan'],loc='lower left')
# 將 l1  加入至目前的 Axes
plt.gca().add_artist(l1)
plt.show()

