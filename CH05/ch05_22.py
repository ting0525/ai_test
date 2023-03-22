import tensorflow as tf
from tensorflow import keras
y_true = tf.constant([[0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 0]])
y_pred = tf.constant([[0.2, 0.8, 0],
                      [0.1, 0.1, 0.8],
                      [0.9, 0.05, 0.05]])
loss3 = keras.losses.categorical_crossentropy(y_true,y_pred)
loss4 = tf.reduce_mean(keras.losses.categorical_crossentropy(y_true,y_pred))
print(loss3)
print(loss4)