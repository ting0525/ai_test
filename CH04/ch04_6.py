import tensorflow as tf
import numpy as np

print(tf.constant(np.pi, dtype= tf.float32))  # 利用 tf.float32 保存 pi 常量
print(tf.constant(np.pi, dtype= tf.float64))  # 利用 tf.float64 保存 pi 常量