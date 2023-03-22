import tensorflow as tf

# 利用 tf 創造標量
scalarValue = tf.constant(3.2)
# 利用 tf 創造向量
vectorValue = tf.constant([1.2,2.4,3.6])
# 利用 tf 創造矩陣
materixValue = tf.constant([[1.2,2.4,3.6],
                            [4.8,5.0,6.2]])
# 利用 tf 創造多維矩陣
multiMaterixValue = tf.constant([[[1.2,2.4],[3.6,4.8]],
                                 [[2.1,4.2],[6.3,8.4]]])
print("scalarValue =",scalarValue,"    shape :",
      scalarValue.shape)
print("vectorValue =",vectorValue,"    shape :",
      vectorValue.shape)
print("materixValue =",materixValue,"    shape :",
      materixValue.shape)
print("multiMaterixValue =",multiMaterixValue,"    shape :",
      multiMaterixValue.shape)

numValue = vectorValue.numpy()
print(numValue)

Degree3 = tf.constant([[[1.2,2.4],[3.6,4.8]],[[2.1,4.2],[6.3,8.4]]])
print(Degree3.ndim)   # 印出 Degree3 張量的階數
