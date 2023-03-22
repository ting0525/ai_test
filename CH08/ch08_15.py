import tensorflow as tf
# 創建兩個梯度張量
w1=tf.random.uniform([3,3],minval=0,maxval=4)
w2=tf.random.uniform([3,3],minval=0,maxval=4)
print("未裁剪時 w1 :",w1)
print("未裁剪時 w2 :",w2)
# 計算 global norm
global_norm=tf.math.sqrt(tf.norm(w1)**2+tf.norm(w2)**2)
print("global_norm = ",global_norm)
# 計算並傳回 global_norm 計算 w1 與 w2 的範數是否大於2 ,
# 如果大於就傳回裁剪過的梯度張量
(cw1,cw2),global_norm_1 = tf.clip_by_global_norm([w1,w2],3)
print("global_norm_1 = ",global_norm_1)
print("裁剪時 w1 :",cw1)
print("裁剪時 w2 :",cw2)
# 計算裁剪後的張量組的 global norm
global_norm2 = tf.math.sqrt(tf.norm(cw1)**2+tf.norm(cw2)**2)
print("global_norm2 = ",global_norm2)

