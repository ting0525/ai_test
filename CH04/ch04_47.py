import tensorflow as tf

train = tf.random.normal([1,1,5,240,1,120])
train0 = tf.squeeze(train)
train1 = tf.squeeze(train,[0,4])
print(train.shape)
print(train0.shape)
print(train1.shape)