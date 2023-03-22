import tensorflow as tf

Company = tf.random.uniform([3,10,5],maxval=50,dtype=tf.int32)
number = tf.gather(Company,[0,2,4],axis=1)
print(number.shape)