from tensorflow.keras.datasets import reuters
# 下載最常見的多少字
num_word = 10000
(train_data,train_label),(test_data,test_label) = \
    reuters.load_data(num_words=num_word)
print("train_data.shape",train_data.shape)
print("train_label.shape",train_label.shape)
print("test_data.shape",test_data.shape)
print("test_label.shape",test_label.shape)
print("train_data[0] :",train_data[0])
print("train_label[0] :",train_label[0])

Index_of_word = reuters.get_word_index()
youIndex = Index_of_word["you"]
print("'you' index = ",youIndex)

All_Word_Map = dict([(value, key) for (key, value)
                     in Index_of_word.items()])
print(youIndex,'=',All_Word_Map[youIndex])

content_of_first = [All_Word_Map.get(i-3,"?")
                    for i in train_data[0]]
# 將 content_of_first 清單以空白字元連接起來
news_of_first = " ".join(content_of_first)
print(news_of_first)

# 資料預處理
from tensorflow.keras.preprocessing import sequence
# 將原始新聞長度裁剪成固定長度
wordMaxNum = 200
train_data_new = sequence.pad_sequences(train_data,
                                        maxlen = wordMaxNum)
test_data_new = sequence.pad_sequences(test_data,
                                       maxlen = wordMaxNum)
print(train_data_new.shape)
print(test_data_new.shape)

import tensorflow as tf
# 定義類別數目
num_classes = 46
One_hot_Train = tf.one_hot(train_label,depth=num_classes)
One_hot_Test = tf.one_hot(test_label,depth=num_classes)

from tensorflow.keras.layers import LSTM
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential()
model.add(layers.Embedding(num_word,
                           output_dim=200,
                           input_length= wordMaxNum))
model.add(LSTM(128,dropout=0.5,return_sequences=True))
model.add(LSTM(128,dropout=0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
print(model.summary())

import matplotlib.pyplot as plt
batch_sizes = 32
epochs = 50
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
hist = model.fit(train_data_new,One_hot_Train,epochs=epochs,
                 batch_size=batch_sizes,verbose=2,
                 validation_split=0.2)