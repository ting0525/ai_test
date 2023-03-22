import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def ReBrightnessImage(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)  # RGBA
    # 調正亮度 : delta 建議 0 ~ 1 之間
    img = tf.image.adjust_brightness(img,delta = 0.5)
    showimage = np.asarray(img.numpy(),dtype='uint8')
    plt.figure(1)
    plt.imshow(showimage)
    plt.show()

ReBrightnessImage('test.jpg')

