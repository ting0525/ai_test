import tensorflow as tf

a1 = tf.constant([[3.0,3.0,3.0],
                  [3.0,3.0,3.0],
                  [3.0,3.0,3.0]])
a2 = tf.constant([[1.5,1.5,1.5],
                  [1.5,1.5,1.5],
                  [1.5,1.5,1.5]])

# 乘法
mul1 = tf.multiply(a1,a2)
mul2 = a1 * a2  # 與 multiply 等價
print(mul1,mul2,sep='\n')
# 除法
div1 = tf.divide(a1,a2)
div2 = a1 / a2  # 與 divide 等價
print(div1,div2,sep='\n')

