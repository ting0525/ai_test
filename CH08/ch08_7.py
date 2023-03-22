import tensorflow as tf
import tensorflow.keras as keras

batch_size = 5
time_step = 10
embedding_dim = 20
# 要輸入的資料 (5個句子, 每個句子10個單詞)
data = tf.random.normal(shape=[batch_size,time_step,embedding_dim])
hidden_dim = 32  # 隱藏層維度
#  設定 h0 一開始的值 (設定初始化狀態向量)
h0 = tf.random.normal(shape=[batch_size,hidden_dim])
# 建立隱藏層維度為 32 的 SimpleRNNCell
simpleRNNCell = keras.layers.SimpleRNNCell(hidden_dim)
h = h0
out = 0
# 完成一個完整序列的前向計算
for xt in tf.unstack(data, axis=1):
    out, h = simpleRNNCell(xt,h)
# 最終的輸出可以集合每個時間步的輸出, 也可以取最後的時間步的結果
out = out


