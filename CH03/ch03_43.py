import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
axis = fig.gca(projection='3d')
# 產生格點資料
x, y, z = np.meshgrid(np.arange(-1.0, 1, 0.2),
                      np.arange(-1.0, 1, 0.2),
                      np.arange(-1.0, 1, 0.5))
# 產生向量資料
u = np.sin(x)
v = -np.cos(y)
w = np.sin(z)
# 繪製向量場
axis.quiver(x, y, z, u, v, w, length=0.2, color = 'red', normalize=True)
# 設置 x,y,z 軸標籤
axis.set_xlabel("X", fontsize=14)
axis.set_ylabel("Y", fontsize=14)
axis.set_zlabel("Z", fontsize=14)
plt.show()

