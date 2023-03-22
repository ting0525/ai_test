import tensorflow as tf

a = tf.constant([[1,2,3],
                 [4,5,6]])
b = tf.constant([[7,8,9],
                 [10,11,12]])
c = tf.stack([a,b],axis = 0)
d = tf.stack([a,b],axis = 1)
e = tf.stack([a,b],axis = 2)
print(c)
print(d)
print(e)