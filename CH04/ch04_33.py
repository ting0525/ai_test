import tensorflow as tf

#  平方計算
x = tf.range(3)
y = tf.pow(x,2)
z = x**2
print(y)
print(z)