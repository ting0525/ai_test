import tensorflow as tf

a1 = tf.constant([[3.0,3.0,3.0],
                  [3.0,3.0,3.0],
                  [3.0,3.0,3.0]])
a2 = tf.constant([[1.5,1.5,1.5],
                  [1.5,1.5,1.5],
                  [1.5,1.5,1.5]])
# 加法 :
add1 = tf.add(a1,a2)
add2 = a1 + a2 # 與 add 等價
print(add1,add2,sep='\n')
# 減法 :
sub1 = tf.subtract(a1,a2)
sub2 = a1 - a2  # 與 subtract 等價
print(sub1,sub2,sep='\n')

