import tensorflow as tf
from tensorflow.keras import layers
# 建立輸入資訊 (批次大小, 時間步, 隱藏層數目)
inputs = tf.random.normal([2, 5, 8])
# 創建 LSTM, 輸出空間為度為 32
LSTM = layers.LSTM(32, return_sequences=True)
output = LSTM(inputs)
print("output.shape",output.shape)
