import tensorflow as tf
from tensorflow.keras import layers  # 導入 layer 類

class netModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 創建四個全連接網路
        self.fc1 = layers.Dense(256, activation=tf.nn.relu)
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        self.fc3 = layers.Dense(64, activation=tf.nn.relu)
        self.fc4 = layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        # 撰寫網路各層順序
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)
        return out

input = tf.random.normal([3,784])
myModel = netModel()   # 建立網路
out = myModel(input)
print(myModel.summary())   # 印出網路架構訊息
print(out.shape)  # 將輸出維度大小印出

