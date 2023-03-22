import matplotlib.pyplot as plt
import numpy as np
# 建立 x 範圍
x = np.linspace(-5,5)
plt.subplot(2,2,1)
plt.plot(x,x)
plt.gca().title.set_text('y=x')
# 右上塊子圖
plt.subplot(2,2,2)
plt.plot(x,np.log(x))
plt.gca().title.set_text('y=log(x)')
# 左下塊子圖
plt.subplot(2,2,3)
plt.plot(x,-x)
plt.gca().title.set_text('y=-x')
# 右下塊子圖
plt.subplot(2,2,4)
plt.plot(x,x**2)
plt.gca().title.set_text('y=x**2')
# 改善個子圖間距
plt.tight_layout()
plt.show()

