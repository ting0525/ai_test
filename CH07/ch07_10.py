import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

CNNModel = Sequential()
# 加入一個卷積層
CNNModel.add(Conv2D(filters=1,
                    kernel_size=(3,3),
                    kernel_initializer= tf.keras.initializers.ones(),
                    input_shape=(5,5,1),
                    activation='relu'))  # 設定激勵函數

# 輸入 [1組資料, 高為5, 寬為5, 通道數為 1]
x = tf.random.normal([1,5,5,1])
# 前項計算
out = CNNModel(x)
print(out)