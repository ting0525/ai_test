import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def Resize_CutImage2(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)  # RGBA
    # 裁減圖形, 比例為原圖的 0.7
    img = tf.image.central_crop(img,0.7)
    showimage = np.asarray(img.numpy(),dtype='uint8')
    plt.figure(1)
    plt.imshow(showimage)
    plt.show()

Resize_CutImage2('test.jpg')
