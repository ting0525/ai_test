import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten

CNNModel = Sequential()
# 加入一個展平層
CNNModel.add(Flatten())
# 輸入 [3組資料, 高為5, 寬為5, 通道數為 3]
x = tf.random.normal([3,5,5,3])
# 前項計算
out = CNNModel(x)
print(out.shape)
