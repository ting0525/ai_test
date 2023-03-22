import tensorflow as tf

A = tf.random.normal([3,20,4])  # 模擬 A 公司的三個部門
B = tf.random.normal([4,20,4])  # 模擬 B 公司的四個部門
C = tf.concat([A,B],axis=0)
print(C.shape)