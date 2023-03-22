import tensorflow as tf
a1 = tf.range(10)  # 創建一個 0~10(不包含10),步長為1的序列
a2 = tf.range(10,delta=2)  # 創建一個 0~10(不包含10),步長為2的序列
a3 = tf.range(10,30,delta=2)   # 創建一個 0~30(不包含30),步長為2的序列
print(a1)
print(a2)
print(a3)