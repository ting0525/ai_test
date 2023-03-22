import tensorflow as tf

train = tf.random.normal([5,240,120])
train0 = tf.expand_dims(train,0)
train1 = tf.expand_dims(train,1)
train_1 = tf.expand_dims(train,-1)
print(train.shape)
print(train0.shape)
print(train1.shape)
print(train_1.shape)