import tensorflow as tf
from tensorflow import keras

out = tf.random.normal([2,10])  # 創造網路輸出的值
yTrue = tf.constant([3,7])     # 創造真實值
yTrueOnehot = tf.one_hot(yTrue,depth=10)   # 轉換成 Onehot 形式
print(yTrueOnehot)
loss = keras.losses.MSE(yTrueOnehot,out)
print(loss)

