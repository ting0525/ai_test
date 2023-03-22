import numpy as np
import matplotlib.pyplot as plt
# 得到目前畫布
fig = plt.figure()
# 得到目前繪圖區, 並改成 3d 投影
ax = fig.gca(projection='3d')
# 產生 3D 座標資料
z = np.linspace(0, 20, 150)
x1 = np.sin(z)  # 利用 z 值產生 x 座標
y1 = np.cos(z)
# 繪製 3D 曲線
ax.plot(x1, y1, z, color= 'green', label='3D Curve')
# 產生 3D 座標資料
x2 = x1 + 0.1 * np.random.randn(150)
y2 = y1 + 0.1 * np.random.randn(150)
# 繪製 3D 座標點
ax.scatter(x2, y2, z, c=z, cmap='jet', label='Curve Points')
# 顯示資料的label名稱
ax.legend()
# 顯示圖形
plt.show()

