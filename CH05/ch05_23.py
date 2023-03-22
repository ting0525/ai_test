import tensorflow as tf
y_true =tf.constant([1,2,0])  # 等數標籤值
y_pred = tf.constant([[0.05, 0.95, 0],
                      [0.1, 0.1, 0.8],
                      [0.9, 0.05, 0.05]])
loss5 = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
loss6 = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
print(loss5)
print(loss6)