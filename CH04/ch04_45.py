import tensorflow as tf

x = tf.random.normal([5,20,4])
y = tf.unstack(x, axis=0)   # 根據 axis=0 做拆分
print(len(y))
print(y[0].shape)