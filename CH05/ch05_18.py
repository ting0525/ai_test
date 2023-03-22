import tensorflow as tf

logits1 = tf.constant([[9.,0.],
                      [2.,8.],
                      [1.,9.],
                      [3.,7.]])
labels = tf.constant([0,1,1,1])
one_hot_labels = tf.one_hot(labels, depth=2)
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=one_hot_labels,logits=logits1))
print(loss1)