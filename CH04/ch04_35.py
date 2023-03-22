import tensorflow as tf

x = tf.range(3)
x = tf.cast(x, dtype= tf.float32)  # 整數轉浮點數
y = tf.square(x)  # 算出 x 的平方
z = tf.sqrt(y)    # 算出 y 的平方根
print(y)
print(z)