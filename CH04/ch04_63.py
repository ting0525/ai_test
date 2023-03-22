import tensorflow as tf

a1 = tf.ones([3,3])
a2 = tf.zeros([3,3])
cond = tf.constant([[True,False,True],
                    [False,True,False],
                    [True,False,True]])
a3 = tf.where(cond,a1,a2)
print(a3)