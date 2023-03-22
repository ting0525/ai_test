import tensorflow as tf

str = tf.constant([['Hello'],['Python']])
substr = tf.strings.substr(str, pos=1,len=3)
print(substr)