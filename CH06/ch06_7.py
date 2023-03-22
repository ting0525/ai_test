import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
def ResizeImage(x):
    image = tf.io.read_file(x)
    # 將圖像使用JPEG的格式解碼從而得到圖像對應的三維矩陣。
    image = tf.image.decode_jpeg(image, channels=3)  # RGBA
    print("原始影像大小 :",image.shape)  # 顯示原始大小
    image = tf.image.resize(image, [128,128])  # 對影像縮小
    image = np.asarray(image.numpy(),dtype='uint8')
    plt.imshow(image)
    plt.show()

ResizeImage('test.jpg')