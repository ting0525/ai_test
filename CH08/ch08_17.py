import tensorflow as tf
from tensorflow.keras import layers
# input 格式 =[句子數目, 每句單詞數目, 隱藏層數目]
input = tf.random.normal([2,10,50])
state = [tf.zeros([2,32]),tf.zeros([2,32])]
# 創建 LSTMCell, 輸出空間為度為 32
Cell = layers.LSTMCell(32)
for xt in tf.unstack(input, axis=1):
    output,state = Cell(xt,state)
    print(id(output))
