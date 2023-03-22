import tensorflow as tf

x = tf.random.normal([5,20,4])
# 沿著 dim = 0切割成 5份
y = tf.split(x, num_or_size_splits=5,axis=0)
print(len(y))  # 印出 y 切割後的份數
print(y[0].shape)  # 印出第一份的張量大小
