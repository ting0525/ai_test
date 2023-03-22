import tensorflow as tf

a = tf.random.normal([5,5])
z = tf.zeros([5,5])
print(a)
mask = a>0
print(mask)
e = tf.where(mask,a,z)
print(e)