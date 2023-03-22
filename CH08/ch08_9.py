import tensorflow as tf
from tensorflow.keras import layers
# 建立兩個句子, 每個句子10個單詞
word = tf.random.normal([2,10,20])
x0 = word[:,0,:]   # 取第一個時間戳的輸入 x0
# 構建 2 個 Cell,下面為 cell_0,上面為 cell_1，狀態向量長度都為 16
cell_0 = layers.SimpleRNNCell(16)
cell_1 = layers.SimpleRNNCell(16)
h0 = [tf.zeros([2,16])]   # cell0 的初始狀態向量
h1 = [tf.zeros([2,16])]   # cell1 的初始狀態向量
# 保存第一層所有時間步上的輸出
outLevel1 = []
# 計算第一層的所有時間步的前向計算
for x1 in tf.unstack(word, axis=1):
    # xt 作為輸入，輸出為 out0
    out0, h0 = cell_0(x1, h0)
    outLevel1.append(out0)

# 計算第二層的所有時間步的前向計算
for m1 in outLevel1:
    # xt 作為輸入，輸出為 out1
    out1, h1 = cell_1(m1, h1)