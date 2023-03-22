import tensorflow as tf

A = tf.constant([[1,2,-1],
                 [3,1,0]])
B = tf.constant([[1,3],
                 [2,2],
                 [3,1]])
mul = tf.matmul(A,B)
print(mul)