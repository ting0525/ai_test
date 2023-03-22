import tensorflow as tf
# 利用 tf 創造張量 w
w = tf.Variable([3.],dtype= tf.float32)
# 利用 tf 創造張量 x
x = tf.Variable([2.],dtype= tf.float32)
# 利用 tf 創造張量 b
b = tf.Variable([5.],dtype= tf.float32)
# o = sigmoid(w*x+b)
print(w*x+b)
o = tf.sigmoid(w*x+b)
print(o)


