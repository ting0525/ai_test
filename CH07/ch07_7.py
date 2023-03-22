import tensorflow as tf

# 輸入 [1組資料, 高為7, 寬為7, 通道數為 1]
x = tf.random.normal([1,7,7,1])
# 建立[高為3,寬為3,輸入通道為1,兩個 filter]
filter = tf.random.normal([3,3,1,2])
# 設定上下步伐為2, 且上下左右各填充4行列的 0
out = tf.nn.conv2d(input=x,filters=filter,strides=[1,2,2,1],
                   padding=[[0,0],[4,4],[4,4],[0,0]])
print(out.shape)