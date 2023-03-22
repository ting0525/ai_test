import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def flip_left_right(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)  # RGBA
    # 随機水平翻轉
    img = tf.image.random_flip_left_right(img)
    showimage = np.asarray(img.numpy(),dtype='uint8')
    plt.figure(1)
    plt.imshow(showimage)
    plt.show()

flip_left_right('test.jpg')
