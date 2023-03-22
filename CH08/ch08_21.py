import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
vocab_size = 10000
maxlen = 100
batch_size = 32
(trainData, trainLabel), (testData, TestLabel) = \
    imdb.load_data(num_words=vocab_size)
# 將訓練集與測試集的字串大小固定成 100 大小
trainData = sequence.pad_sequences(trainData, maxlen=maxlen)
testData = sequence.pad_sequences(testData, maxlen=maxlen)
print('trainData shape:', trainData.shape)
print('testData shape:', testData.shape)


from tensorflow.keras.layers import LSTM
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential()
model.add(layers.Embedding(vocab_size,
                           output_dim=120,
                           input_length= maxlen))
model.add(LSTM(64,dropout=0.5,return_sequences=True))
model.add(LSTM(64,dropout=0.5))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
hist = model.fit(trainData, trainLabel,
                 epochs=15,
                 batch_size=batch_size
                 )

loss,accuracy = model.evaluate(testData,TestLabel)
print("測試集的正確率 = ",accuracy)

