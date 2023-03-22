import tensorflow as tf
i1 = tf.constant([True,True,False,False])
b1 = tf.cast(i1,tf.int32)
print(b1)