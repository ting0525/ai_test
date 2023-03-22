import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5,5)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(x)
y4 = np.log(x)
# 讓四張小圖都有名稱
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                    figsize=(12, 10),dpi=72)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=0.3, hspace=0.3)
# 直接用小圖名稱代替
ax1.set_title("Sin", fontsize=16)
ax1.plot(x, y1)
ax2.set_title("Cos", fontsize=16)
ax2.plot(x, y2)
ax3.set_title("Exponential", fontsize=16)
ax3.plot(x, y3)
ax4.set_title("log", fontsize=16)
ax4.plot(x, y4)
plt.show()




