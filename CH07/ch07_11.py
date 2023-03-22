from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import tensorflow as tf

CNNModel = Sequential()
# 加入一個卷積層
CNNModel.add(Conv2D(filters=1,
                    kernel_size=(3,3),
                    kernel_initializer= tf.keras.initializers.ones(),
                    input_shape=(7,7,1)))

CNNModel.add(Conv2D(32, (3,3),
                    kernel_constraint = tf.keras.constraints.max_norm(3),
                    bias_constraint = tf.keras.constraints.max_norm(3)))


# 得到權重值
print(CNNModel.get_weights())