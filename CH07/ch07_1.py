from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

model = Sequential([
    layers.Flatten(input_shape=(28, 28)),   # 將輸入資料從 28x28 攤平成 784
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # output 為 10 個 class
])
print(model.summary())
