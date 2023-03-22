import tensorflow as tf

arr = tf.range(8)
arrA = tf.reshape(arr,[2,4])
print(arrA)
arrB = tf.gather(arrA,[3,2,1,0],axis=1)
print(arrB)