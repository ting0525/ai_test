import tensorflow as tf

str1 = tf.constant("Hello, Tensorflow ")
str2 = tf.convert_to_tensor("Hello, Python")
print('str1 :',str1,' shape :', str1.shape)   # 印出字串內容與形狀
print('str2 :',str2,' shape :', str2.shape)