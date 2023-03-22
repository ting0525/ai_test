import tensorflow as tf
logits2 = tf.constant([[9.,0.,2.],
                       [2.,8.,1.],
                       [1.,9.,3.],
                       [3.,1.,7.]])
labels = tf.constant([0,1,1,2])
loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels,logits=logits2))
print(loss2)
