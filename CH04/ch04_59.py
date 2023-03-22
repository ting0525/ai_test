import tensorflow as tf

Company = tf.random.uniform([3,10,5],maxval=50,dtype=tf.int32)
mask = [True,False,True,False,True,False,False,False,False,False]
number = tf.boolean_mask(Company,mask,axis=1)
print("number.shape : ", number.shape)