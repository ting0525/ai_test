import matplotlib.pyplot as plt
import numpy as np
# 左上塊子圖
x = np.linspace(-5,5)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(x)
y4 = np.log(x)

# 建立兩列兩行的子圖
fig,ax = plt.subplots(2,2)
ax[0, 0].plot(x, y1)  # 左上圖
ax[0, 0].text(0., 0., str((0, 0)), fontsize=18, ha='center')
ax[0, 1].plot(x, y2)  # 右上圖
ax[0, 1].text(0., 0., str((0, 1)), fontsize=18, ha='center')
ax[1, 0].plot(x, y3)  # 左下圖
ax[1, 0].text(0., 75., str((1, 0)), fontsize=18, ha='center')
ax[1, 1].plot(x, y4)  # 右下圖
ax[1, 1].text(2.5, -0.5, str((1, 1)), fontsize=18, ha='center')
# 設定標頭
ax[0, 0].set_title("Sin")
ax[0, 1].set_title("Cos")
ax[1, 0].set_title("Exponential")
ax[1, 1].set_title("log")

fig.tight_layout()
plt.show()






