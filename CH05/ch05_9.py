#ch05-8的改良版 使用Dense來實現全連接
import tensorflow as tf
from tensorflow.keras import layers  # 導入 layer 類

fc1 = layers.Dense(256,activation=tf.sigmoid)  # 隱藏層 1
fc2 = layers.Dense(128,activation=tf.sigmoid)  # 隱藏層 2
fc3 = layers.Dense(64,activation=tf.sigmoid)  # 隱藏層 3
fc4 = layers.Dense(10,activation=None)  # 輸出層

#輸入層、隱藏層、輸出層的全連接
x = tf.random.normal([3,784])
h1 = fc1(x)  
h2 = fc2(h1)
h3 = fc3(h2)
out = fc4(h3)
print(out.shape)   # 輸出網路輸出維度大小
