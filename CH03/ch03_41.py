import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
axis = fig.gca(projection='3d')
# 設置 3D 線框圖點資訊
x = np.arange(-3.0, 3.0, 0.2)
y = np.arange(-3.0, 3.0, 0.2)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2
# 繪製3D線框
surface = axis.plot_wireframe(X, Y, Z, rstride=1,
                              cstride=1, color ='green')
# 設置圖表訊息
plt.title("Z = X**2 - Y**2", fontsize=14)
# 設置 x,y,z 軸標籤
axis.set_xlabel("X", fontsize=14)
axis.set_ylabel("Y", fontsize=14)
axis.set_zlabel("Z", fontsize=14)
plt.show()