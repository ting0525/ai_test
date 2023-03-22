import tensorflow as tf

# 創建 W,b 張量
x = tf.random.normal([3,784])
# 隱藏層 1權重與偏移值設定
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 隱藏層 2權重與偏移值設定
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 隱藏層 3權重與偏移值設定
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 輸出層權重與偏移值設定
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))
# 前向計算  # '@'為矩陣乘法符號
o1 = x@w1 + b1  
s1 = tf.nn.sigmoid(o1)
o2 = s1@w2 + b2
s2 = tf.nn.sigmoid(o2)
o3 = s2@w3 + b3
s3 = tf.nn.sigmoid(o3)
o4 = s3@w4 + b4

print(o4.shape)
