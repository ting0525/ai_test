import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
# 建立輸入資訊 (批次大小, 時間步, 隱藏層數目)
inputs = tf.random.normal([2, 5, 8])
model = keras.Sequential()
# 加入兩層 LSTM
model.add(layers.LSTM(32, return_sequences=True))
model.add(layers.LSTM(16))
# 呼叫模型名稱就可以完成所有的的層數與時間步的前向運算
output = model(inputs)
print("output.shape",output.shape)

