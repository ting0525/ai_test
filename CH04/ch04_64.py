import tensorflow as tf

a4 = tf.random.normal([3,3])
print(a4)
mask = a4>0
print('mask :',mask)
# 找出 True 的索引值
indices = tf.where(mask)
print('indices :',indices)