import tensorflow as tf

s = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
s1 = tf.tile(s, [2, 3])
print(s1)