import tensorflow as tf
from tensorflow.keras import layers
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 定義文件檔 (前七個是正情緒, 後七個是負情緒)
docs1 = ['perfect','excellent','dreams come true',
        'best Wishes','nice work','fine','good for You',
        'chin up','sad','bad mood','Poor effort',
        'down in the dump','feel blue','very bad']
# 定義文件檔的標籤 (正情緒與負情緒詞)
labels = array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

# 定義詞匯的數量
vocab_size = 70
# 利用 one_hot()函數來做整數編碼
encoded_docs = [one_hot(text, vocab_size) for text in docs1]
print(encoded_docs)

# 最大句子的長度
maxlen = 4
padded_docs = pad_sequences(encoded_docs, maxlen=maxlen, padding='post')
print(padded_docs)

model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, 8, input_length=maxlen))
model.add(layers.Flatten())
# 加上一般的完全連接層(Dense)
model.add(layers.Dense(1, activation='sigmoid'))
# 印出模型
print(model.summary())

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
# 執行模型
model.fit(padded_docs, labels, epochs=50, verbose=0)
# 計算損失與正確率
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))