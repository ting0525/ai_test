import tensorflow as tf

Company = tf.random.uniform([2,5,3],maxval=50,dtype=tf.int32)
number = tf.gather_nd(Company,[[0,0],[0,1],[1,2],[1,3]])
print(number)