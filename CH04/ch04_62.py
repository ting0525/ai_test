import tensorflow as tf

Company = tf.random.uniform([2,5,3],maxval=50,dtype=tf.int32)
mask = [[True,True,False,False,False],
        [False,False,True,True,False]]
number = tf.boolean_mask(Company,mask)
print(number)