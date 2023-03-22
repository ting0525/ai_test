import tensorflow as tf

str = tf.constant('I like Python Language very much')
substr = tf.strings.substr(str, pos=2,len=11)
print(substr)