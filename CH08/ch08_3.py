import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import imdb
# 找出最常出現的2000個單詞
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=2000)
print("train_data shape :", train_data.shape)
print("train_labels shape :", train_labels.shape)
print("test_data shape :", test_data.shape)
print("test_labels shape :", test_labels.shape)

print(train_data[0])
print(train_labels[0])

# data 是將單詞對映到整數索引的字典
data = imdb.get_word_index()
print(data)
# 鍵值顛倒，將整數索引對映到單詞
word_map = dict([(value,key) for (key,value) in data.items()])
words = []
for word_index in train_data[0]:
    words.append(word_map.get(word_index-3,'?'))
print(" ".join(words))

print("len(train_data[0]) :",len(train_data[0]))
print("len(train_data[1]) :",len(train_data[1]))

from tensorflow.keras.preprocessing import sequence
max_words = 100
train_data_new = sequence.pad_sequences(train_data,maxlen=max_words)
test_data_new = sequence.pad_sequences(test_data,maxlen=max_words)
print(train_data_new.shape)
print(test_data_new.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense,Flatten,Dropout
vocab_size = 2000
model = Sequential()
model.add(Embedding(vocab_size,16,input_length=max_words))
model.add(Flatten())
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(1, activation=tf.nn.sigmoid))
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_history = model.fit(train_data_new, train_labels,
                          batch_size=128, epochs=5, verbose=2,
                          validation_split=0.2)


loss,accuracy = model.evaluate(test_data_new,test_labels)
print("測試集的正確率 = ",accuracy)



max_index = max(max(sequence) for sequence in train_data)
print(max_index)
