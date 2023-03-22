import tensorflow as tf

Mz1 = tf.zeros([2,3])
Mz2 = tf.zeros_like(Mz1)
Mo1 = tf.ones([2,3])
Mo2 = tf.ones_like(Mo1)
print('Mz1 :',Mz1)
print('Mz2 :',Mz2)
print('Mo1 :',Mo1)
print('Mo2 :',Mo2)