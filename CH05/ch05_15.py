# 匯入 Keras 提供的序列式模型類別
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# 匯入Keras的mnist模組
from tensorflow.keras.datasets import mnist
(train_Data, train_Label), (test_Data, test_Label) = mnist.load_data()

model = Sequential([
    layers.Flatten(input_shape=(28, 28)),   # 將輸入資料從 28x28 攤平成 784
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # output 為 10 個 class
])

# model 每層定義好後需要經過 compile
# sparse_categorical_crossentropy 的標籤是 integer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 將建立好的 model 去 fit 我們的 training data
model.fit(train_Data, train_Label, epochs=10)
# 利用 test_Data 去進行模型評估
# verbose = 2 為每個 epoch 輸出一行紀錄
model.evaluate(test_Data, test_Label, verbose=2)

