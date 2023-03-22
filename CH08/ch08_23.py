from tensorflow.keras.datasets import reuters
import tensorflow as tf
physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], enable = True)
# 下載最常見的多少字
num_word = 10000
(train_data,train_label),(test_data,test_label) = \
    reuters.load_data(num_words=num_word)

# 資料預處理
from tensorflow.keras.preprocessing import sequence
# 將原始新聞長度裁剪成固定長度
wordMaxNum = 200
train_data_new = sequence.pad_sequences(train_data,
                                        maxlen = wordMaxNum)
test_data_new = sequence.pad_sequences(test_data,
                                       maxlen = wordMaxNum)

# 定義類別數目
num_classes = 46
One_hot_Train = tf.one_hot(train_label,depth=num_classes)
One_hot_Test = tf.one_hot(test_label,depth=num_classes)

from tensorflow.keras.layers import GRU
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential()
model.add(layers.Embedding(num_word,
                           output_dim=250,
                           input_length= wordMaxNum))
model.add(GRU(128,dropout=0.5,return_sequences=True))
model.add(GRU(64,dropout=0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
print(model.summary())

batch_sizes = 128
epochs = 40
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
hist = model.fit(train_data_new,One_hot_Train,epochs=epochs,
                 batch_size=batch_sizes,verbose=2,
                 validation_split=0.2)

loss,accuracy = model.evaluate(test_data_new,One_hot_Test)
print("測試集的正確率 = ",accuracy)

