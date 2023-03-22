import tensorflow as tf

i1 = tf.constant([1, 2, 3, 4, 5])
f1 = tf.cast(i1, dtype=tf.float32)   # int32 轉換為 float32
print(i1)
print(f1)