import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
def RotateImage(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.rot90(img, 3)  # 順時針旋轉 90 度三次
    showimage = np.asarray(img.numpy(),dtype='uint8')
    plt.figure(1)
    plt.imshow(showimage)
    plt.show()

RotateImage('test.jpg')
