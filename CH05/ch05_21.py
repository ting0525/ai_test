import tensorflow as tf
from tensorflow import keras

y_true = tf.constant([[0, 1],
                      [1, 0]])
y_pred = tf.constant([[0.8, 0.2],
                      [1., 0.]])
loss = keras.losses.binary_crossentropy(y_true,y_pred)
loss1 = tf.reduce_mean(keras.losses.binary_crossentropy(y_true,y_pred))
print(loss)
print(loss1)