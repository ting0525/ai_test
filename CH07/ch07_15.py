import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


CNNModel = Sequential()
# 加入一個全連接層
CNNModel.add(Dense(10))
# 輸入 [3組資料, 高為5, 寬為5, 通道數為 3]
x = tf.random.normal([3,5,5,3])
# 前項計算
out = CNNModel(x)
print(out.shape)


