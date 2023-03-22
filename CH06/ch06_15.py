import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def Resize_CutImage(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)  # RGBA
    #先將影像縮放到 247x247
    img = tf.image.resize(img,[247,247])
    # 再將影像裁剪至 227x227
    img = tf.image.random_crop(img,[227,227,3])
    showimage = np.asarray(img.numpy(),dtype='uint8')
    plt.figure(1)
    plt.imshow(showimage)
    plt.show()

Resize_CutImage('test.jpg')

