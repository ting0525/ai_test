from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

(train_image,train_label),(test_image,test_label)=\
    tf.keras.datasets.fashion_mnist.load_data()

print("train_image.shape :",train_image.shape)
print("train_image.shape :",train_label.shape)

# 把訓練集切分成 80% 訓練集, 20% 驗證集
train_data, Valid_data, train_labelNew,  Valid_LabelNew = \
    train_test_split(train_image,train_label,test_size=0.2)

print("train_data.shape :",train_data.shape)
print("Valid_data.shape :",Valid_data.shape)

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
              loss= 'sparse_categorical_crossentropy',
              metrics=['acc'])
history = model.fit(train_data,train_labelNew,
                    # 驗證集可以在這邊設定
                    validation_data=0.2, # 訓練集切出20%給驗證集
                    epochs=500,
                    batch_size=128,  # 設定批次大小
                    shuffle=True)    # 是否打散