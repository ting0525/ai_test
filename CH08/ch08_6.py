import tensorflow as tf
import tensorflow.keras as keras

batch_size = 5
time_step = 10
embedding_dim = 20
# 要輸入的資料
data = tf.random.normal(shape=[batch_size,time_step,embedding_dim])
hidden_dim = 32  # 隱藏層維度
#  設定 h0 一開始的值
h0 = tf.random.normal(shape=[batch_size,hidden_dim])
x0 = data[:,0,:]  # 第一個時間的輸入資料
simpleRNNCell = keras.layers.SimpleRNNCell(hidden_dim)
# 完成一個時間步的運算
out,h1 = simpleRNNCell(x0,[h0])
print("out.shape : ",out.shape)
print("h1[0].shape : ", h1[0].shape)
# 查看 out 與 h1[0] 記憶體存放位址
print("out :",id(out))
print("h1[0] :", id(h1[0]))