import tensorflow as tf

padsymBefor = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
paddings = tf.constant([[1,2], [2,1]])
padsymAfter = tf.pad(padsymBefor, paddings, "symmetric")
print(padsymAfter)