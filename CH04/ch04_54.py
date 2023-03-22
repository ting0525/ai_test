import tensorflow as tf

x = tf.range(9)
y = tf.clip_by_value(x,3,7)
print(y)