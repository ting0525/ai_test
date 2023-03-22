import tensorflow as tf

a = tf.constant(123456789, dtype= tf.int16)
b = tf.constant(123456789, dtype= tf.int32)
print("a = ",a,"dtype = ", a.dtype)
print("b = ",b,"dtype = ", b.dtype)