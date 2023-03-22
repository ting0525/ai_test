import tensorflow as tf

# 輸入 [1組資料, 高為6, 寬為6, 通道數為 3]
x = tf.random.normal([1,6,6,3])
# 建立[高為3,寬為3,輸入通道為3,1個 filter]
filter = tf.random.normal([3,3,3,1])
# 設定上下步伐為1, 且不填充 0
out = tf.nn.conv2d(input=x,filters=filter,strides=[1,1,1,1],
                   padding='VALID')
print(out.shape)
