import tensorflow as tf

x = tf.range(8)
y = tf.maximum(x,5)   # 最大值輸出 5
z = tf.minimum(x,3)   # 最小值輸出 3
a = tf.minimum(tf.maximum(x,3),5)  # 限制值為 3~5之間
print(y)
print(z)
print(a)