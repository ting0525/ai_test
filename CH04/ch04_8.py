import tensorflow as tf

szero = tf.zeros([])  # 創建全為 0 的標量
sone = tf.ones([])   # 創建全為 1 的標量
print('szero :',szero)
print('sone :',sone)

Vzero = tf.zeros([3])   # 創建全為 0 的向量
Vone = tf.ones([3])     # 創建全為 1 的向量
print('Vzero :',Vzero)
print('Vone :',Vone)

Mzero = tf.zeros([2,3])   # 創建全為 0 的矩陣
Mone = tf.ones([2,3])     # 創建全為 1 的矩陣
print('Mzero :',Mzero)
print('Mone :',Mone)