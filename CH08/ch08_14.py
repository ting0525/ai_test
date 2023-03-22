import tensorflow as tf
# 隨機產生梯度值, 並裁剪至指定範圍
w = tf.random.uniform([3,3],minval=0,maxval=4)
value = tf.norm(w,axis=None,ord=2)
# 按范數規定值裁剪
cw = tf.clip_by_norm(w,3)
print("w 的范數 : ",value)
print("裁剪前 :", w)
print("裁剪後 :",cw)

