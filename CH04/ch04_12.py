import tensorflow as tf

tf.random.set_seed(10)
n1 = tf.random.normal([2,3])
tf.random.set_seed(10)
n2 = tf.random.normal([2,3])
print('n1 :',n1)
print('n2 :',n2)