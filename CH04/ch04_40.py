import tensorflow as tf

A1 = tf.random.normal([3,20,4])  # 模擬 三個部門的四項產品
A2 = tf.random.normal([3,20,3])  # 模擬 三個部門的剩下三項產品
ATotal = tf.concat([A1,A2],axis=2)
print(ATotal.shape)