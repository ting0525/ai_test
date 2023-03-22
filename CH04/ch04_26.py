import tensorflow as tf

a = tf.constant([[1,2,3],[4,5,6]])
print(a[1])
print(a[1][0])
print(a[1,0])   # 與 a[1][0] 用法同