import tensorflow as tf
# 隨機產生梯度值, 並裁剪至指定範圍
w = tf.random.uniform([3,3],minval=-2,maxval=2)
cw = tf.clip_by_value(w,-1,1)
print("裁剪前 :", w)
print("裁剪後 :",cw)
