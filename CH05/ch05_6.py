import tensorflow as tf

# 創建 W,b 張量
x = tf.random.normal([3,784])   #3個28*28 (784)的tensor 
w1 = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1)) #隨機生成 #stddev = standard deviation 標準差
b1 = tf.Variable(tf.zeros([10])) #10個輸出層(bias)
o1 = tf.matmul(x,w1) + b1   # 線性變換
o1 = tf.nn.relu(o1)         # 加上激活函數
print(o1.shape)   # 印出輸出大小
