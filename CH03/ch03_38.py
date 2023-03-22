import numpy as np
import matplotlib.pyplot as plt

# 建立 3D 圖形
fig = plt.figure()
# 得到目前 Axes, 也可寫成 ax = plt.axes(projection='3d')
ax = fig.gca(projection='3d')
# 產生兩群 3D 座標資料
z1 = np.random.randint(10,50,50)
x1 = np.random.randint(10,30,50)
y1 = np.random.randint(10,40,50)
z2 = np.random.randint(10,50,50)
x2 = np.random.randint(30,50,50)
y2 = np.random.randint(10,40,50)
# 繪製 3D 座標點
ax.scatter(x1, y1, z1, c=x1, cmap='Oranges', marker='*', label='Cluster-1')
ax.scatter(x2, y2, z2, c=x2, cmap='rainbow', marker='o', label='Cluster-2')
# 設定軸的 label
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
# 顯示資料的label名稱
ax.legend()
# 顯示圖形
plt.show()


