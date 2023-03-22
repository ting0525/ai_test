import tensorflow as tf
# 4個樣本三分類問題，且一個樣本可以同時擁有多類

logits3 = tf.constant([[9.,8.,2.],
                      [2.,8.,1.],
                      [1.,9.,11.],
                      [1.,1.,10.]])
labels = tf.constant([[1,1,0],
                      [0,1,0],
                      [0,1,1],
                      [0,0,1]])
labels = tf.cast(labels, dtype=tf.float32)
loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=labels,logits=logits3))
print(loss3)