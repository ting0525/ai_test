import tensorflow as tf
from tensorflow.keras import layers
# input 格式 =[句子數目, 每句單詞數目, 隱藏層數目]
input = tf.random.normal([2,10,50])
#  給定第一個時間步的輸入資料
xt = input[:,0,:]
state = [tf.zeros([2,32]),tf.zeros([2,32])]
# 創建 LSTMCell, 輸出空間為度為 32
Cell = layers.LSTMCell(32)
output,state = Cell(xt,state)
# 輸出 output, state[0], state[1] 記憶體空間 id
print("id(output) :",id(output))
print("id(state[0]) :",id(state[0]))
print("id(state[1]) :",id(state[1]))