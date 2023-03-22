import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_words = 100
batch_sizes = 128
# 載入 imdb 資料集
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)
# 讓所有的影評資料保持在 100個字
train_data_new = sequence.pad_sequences(train_data,maxlen=max_words)
test_data_new = sequence.pad_sequences(test_data,maxlen=max_words)
db_train = tf.data.Dataset.from_tensor_slices((train_data_new,train_labels))
# 參數 drop_remainder = true 代表當最後一批少於 batch_size元素的情况下就刪除
# 將訓練資料打散
db_train = db_train.shuffle(1000).batch(batch_sizes,drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((test_data_new,train_labels))
db_test = db_test.batch(batch_sizes,drop_remainder=True)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

vocab_size = 10000
model = Sequential()
model.add(layers.Embedding( vocab_size,
                            output_dim=120,
                            input_length= max_words))
# 除了最上面那一層之外, 其他層的輸出都必須做為下一層的輸入
model.add(layers.SimpleRNN(units=64,return_sequences=True,
                           dropout=0.25))
model.add(layers.SimpleRNN(units=64,dropout=0.25))
model.add(layers.Dense(units=1,activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])
history = model.fit(db_train,batch_size=batch_sizes,
                    epochs=10,verbose=2)

loss,accuracy = model.evaluate(db_test)
print("測試集的正確率 = ",accuracy)

