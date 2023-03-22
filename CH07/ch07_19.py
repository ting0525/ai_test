from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 數據讀取
TrainDataGenerator = ImageDataGenerator()
traindata = TrainDataGenerator.flow_from_directory(
                    directory="Cats&Dogs/train",target_size=(224,224))
TestDataGenerator = ImageDataGenerator()
testdata = TestDataGenerator.flow_from_directory(
                    directory="Cats&Dogs/test", target_size=(224,224))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
model = Sequential([
    #　第一組 :兩個 3*3*64 卷積核 + 一個最大池化層
    Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same",
           activation="relu"),
    Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    #　第二組 : 兩個3*3*128卷積核 + 一個最大池化層
    Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    #　第三組 : 三個3*3*56卷積核 + 一個最大池化層
    Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    #　第四組 : 三個3*3*512卷積核 + 一個最大池化層
    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    #　第五組 : 三個3*3*512卷積核 + 一個最大池化層
    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
    MaxPool2D(pool_size=(2,2),strides=(2,2)),
    # 三個全連接層Dense，最後一層用於預測分類。
    Flatten(),
    Dense(units=4096,activation="relu"),
    Dense(units=4096,activation="relu"),
    Dense(units=2, activation="softmax")
])
model.summary()

# 編譯模型, 定義模型優化器， 使用分類交叉熵損失
from tensorflow.keras.optimizers import Adam
import tensorflow.keras
model.compile(optimizer=Adam(lr=0.00001),
              loss = tensorflow.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# 設定監控方法與條件
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# 模型儲存名稱為 vgg16.h5, 監控的評估參數為 val_accuracy
checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_accuracy', verbose=1,
                          save_best_only=True,save_weights_only=False,
                          mode='auto', period=1)
earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0,
                          patience=20, verbose=1, mode='auto')


# 訓練模型並呼叫回調函數
history = model.fit_generator(steps_per_epoch=100,generator=traindata,
                              validation_data= testdata,
                              validation_steps=10,epochs=50,
                              callbacks=[checkpoint,earlystop])

import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show(block=True)

print(history.history.keys())
