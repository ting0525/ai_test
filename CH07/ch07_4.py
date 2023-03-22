import cv2
import numpy as np
import matplotlib.pyplot as plt
# 讀入影像
src = cv2.imread("Lenna.png")
# 設定卷積核 (銳化計算)
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]], dtype="float32")
# 卷積運算
image = cv2.filter2D(src,-1,kernel)
htich = np.hstack((src, image))
plt.imshow(htich)
plt.show()

