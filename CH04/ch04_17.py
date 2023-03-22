import tensorflow as tf

str = tf.constant('I like Python Language')
str1 = tf.strings.split(str,' ')   # 以空格當作分離字元
print(str1)
str = tf.constant('I-like-Python-Language')
str2 = tf.strings.split(str,'-')   # 以'-'當作分離字元
print(str2)