
import tensorflow as tf

int1 = tf.constant(123456789, dtype=tf.int32)
int2 = tf.cast(int1,tf.int16)
print(int2)