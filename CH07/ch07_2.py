import cv2
import numpy as np
import matplotlib.pyplot as plt
# 讀入影像
src = cv2.imread("Lenna.png")
# 設定卷積核 (邊緣計算)
kernel = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]], dtype="float32")
# 卷積運算
image = cv2.filter2D(src,-1,kernel)
htich = np.hstack((src, image))
plt.imshow(htich)
plt.show()