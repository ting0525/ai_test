import tensorflow as tf

A1 = tf.random.normal([20,4])  # 模擬 A部門的四項產品
B1 = tf.random.normal([20,4])  # 模擬 B部門的四項產品
A_B = tf.stack([A1,B1],axis=0)
print(A_B.shape)