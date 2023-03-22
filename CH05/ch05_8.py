# 使用tensor來實現全連接
import tensorflow as tf
from tensorflow.keras import layers  # 導入 layer 類

x = tf.random.normal([3,784])
fc = layers.Dense(10,activation=tf.nn.relu)
out = fc(x)   # 輸入ｘ進行一次前向計算，返回輸出張量
print(out.shape)
print("kernel :",fc.kernel)  # 印出 Dense 的權重矩陣
print("bias :",fc.bias)    # 印出 Dense 的偏移值矩陣
