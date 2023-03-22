import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.datasets import imdb

# 下載數據集及參數設定
# 找出最常出現的2000個單詞
num_word = 2000
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words = num_word)
print("train_data shape :", train_data.shape)
print("train_labels shape :", train_labels.shape)
print("test_data shape :", test_data.shape)
print("test_labels shape :", test_labels.shape)
# data 是將單詞對映到整數索引的字典
data = imdb.get_word_index()
# 鍵值顛倒，將整數索引對映到單詞
word_map = dict([(value,key) for (key,value) in data.items()])
words = []
for word_index in train_data[0]:
    words.append(word_map.get(word_index-3,'?'))
# 將所有單詞以空白字元連接起來
print(" ".join(words))

from tensorflow.keras.preprocessing import sequence
max_words = 100  # 句子最大長度
train_data_new = sequence.pad_sequences(train_data,maxlen=max_words)
test_data_new = sequence.pad_sequences(test_data,maxlen=max_words)

# 定義模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Conv1D,GlobalMaxPool1D
embiding_Dim = 16  # 定義每個詞索引的嵌入向量
filter_Num = 32
kernel_size = 3
model = Sequential()
model.add(Embedding(num_word,embiding_Dim,input_length = max_words))
model.add(Dropout(0.25))
model.add(Conv1D(filter_Num,kernel_size, padding='same',
                 activation='relu',strides=1))
model.add(GlobalMaxPool1D())  # 時序數據最大池化
model.add(Dense(256,activation=tf.nn.relu))  # 全連階層
model.add(Dropout(0.25))
model.add(Dense(1,activation=tf.nn.sigmoid))
print(model.summary())  # 輸出模型訊息

# 模型編譯
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 模型訓練與評估
train_history = model.fit(train_data_new, train_labels,
                          batch_size=128, epochs= 10, verbose=2,
                          validation_split=0.2)

loss,accuracy = model.evaluate(test_data_new,test_labels)
print("測試集的正確率 = ",accuracy)

