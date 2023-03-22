import tensorflow as tf

Company = tf.random.uniform([3,10,5],maxval=50,dtype=tf.int32)
# 取第一,二個部門的成績
Departments = tf.boolean_mask(Company,mask=[True,True,False],axis=0)
print("Departments.shape : ", Departments.shape)