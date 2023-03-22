import tensorflow as tf

Company = tf.random.uniform([3,10,5],maxval=50,dtype=tf.int32)
Departments = tf.gather(Company,[0,1],axis=0)   # 取第一,二個部門的成績
print("Departments.shape : ", Departments.shape)