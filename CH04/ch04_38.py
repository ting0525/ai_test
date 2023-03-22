import tensorflow as tf

x = tf.constant([10.,100.])
y = tf.math.log(x)/tf.math.log(10.)  # 計算 log10 與 log100
print(y)