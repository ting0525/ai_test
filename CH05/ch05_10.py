#ch5-9的改良版 使用sequential來排序全連接的過程
import tensorflow as tf
from tensorflow.keras import layers,Sequential  # 導入 layer 類

x = tf.random.normal([3,784])
# 通過 Sequential 容器封裝為一個網路類
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=None),
])
out = model(x)     # 前向計算得到輸出
print(out.shape)
