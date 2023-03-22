from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping

# 匯入Keras 的 mnist模組
from tensorflow.keras.datasets import mnist
(train_Data, train_Label), (test_Data, test_Label) = mnist.load_data()

model = Sequential([
    layers.Flatten(input_shape=(28, 28)),   # 將輸入資料從 28x28 攤平成 784
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # output 為 10 個 class
])
# 定義訓練的步驟數目
NUM_EPOCHS = 100
# model 每層定義好後需要經過 compile
model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics=['acc',metrics.mse,
                       metrics.sparse_top_k_categorical_accuracy])
# 定義 tf.keras.EarlyStopping 回調函數,
# 並指名監控的對象 => val_sparse_top_k_categorical_accuracy
earlystop_callback = EarlyStopping(
  monitor='val_sparse_top_k_categorical_accuracy', min_delta=0.001,
  patience=1, verbose=1, mode='auto')

# 將建立好的 model 去 fit 我們的 training data
model.fit(train_Data, train_Label,
          validation_split = 0.2,    # 劃分資料集的 20% 作為驗證集用
          epochs=NUM_EPOCHS,callbacks=[earlystop_callback],)
# 利用 test_Data 去進行模型評估
# verbose = 2 為每個 epoch 輸出一行紀錄
model.evaluate(test_Data, test_Label, verbose=2)

