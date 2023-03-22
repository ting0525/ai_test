import tensorflow as tf

x = tf.random.normal([5,20,4])
# 沿著 dim = 0 切割成比例為 2:2:1
y = tf.split(x, num_or_size_splits=[2,2,1],axis=0)
print(len(y))  # 印出 y 切割後的份數
print(y[0].shape)  # 印出第一份的張量大小