import tensorflow as tf

n1 = tf.random.normal([2,3])
n2 = tf.random.normal([2,3],mean=0.0,stddev=1.0)
print(n1)
print(n2)