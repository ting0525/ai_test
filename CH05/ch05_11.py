import tensorflow as tf
from tensorflow.keras import layers,Sequential  # 導入 layer 類

x = tf.random.normal([3,784])
model = Sequential([])  # 創建一個空的網路容器
model.add(layers.Dense(256, activation=tf.nn.relu))  # 加入隱藏層 1
model.add(layers.Dense(128, activation=tf.nn.relu))  # 加入隱藏層 2
model.add(layers.Dense(64, activation=tf.nn.relu))   # 加入隱藏層 3
model.add(layers.Dense(10, activation=None))    # 加入輸出層

out = model(x)
print(out.shape)
print(model.summary())  # 輸出模型各層狀況

