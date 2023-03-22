import tensorflow as tf
from tensorflow.keras import layers
# 創建狀態向量長度為 5 的 SimpleRNN 層
layer = layers.SimpleRNN(5,return_sequences=True)
# 創建輸入的資料
data = tf.random.normal([2, 10, 20])
# 完成一次前向運算
out = layer(data)
print(out.shape)