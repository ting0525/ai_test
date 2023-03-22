import tensorflow as tf
# 矩陣與標量運算
#scalar-tensor操作。
A = tf.constant([1.0,2.0,3.0])
# 加法
A_add2 = A+2
print(A_add2)
# 減法
A_sub2 = A-2
print(A_sub2)
# 乘法
A_mul2 = A*2
print(A_mul2)
# 除法
A_div2 = A/2
print(A_div2)