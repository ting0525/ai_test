import tensorflow as tf

padrefBefor = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
paddings = tf.constant([[1,1], [2,2]])
padrefAfter = tf.pad(padrefBefor, paddings, "REFLECT")
print(padrefAfter)