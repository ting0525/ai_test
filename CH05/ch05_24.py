import tensorflow as tf
import matplotlib.pyplot as plt

(train_image,train_label),(test_image,test_label)=\
    tf.keras.datasets.fashion_mnist.load_data()
print("train_image : ",train_image.shape)
print("train_label : ",train_label.shape)
print("test_image : ",test_image.shape)
print("test_label : ",test_label.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']
# 顯示指定的影像 (這裡顯示九張)
def ShowImage(x,y):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x[i], cmap=plt.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(class_names[y[i]])
    plt.show()

ShowImage(train_image,train_label)


# 對資料集做一個前置處理, 將資料正規到 0~1 之間
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x,y
# 建立模型
def build_model():
    # 線性疊加
    model = tf.keras.models.Sequential()
    # 改變平坦輸入
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # 第一層隱藏層, 包含256個神經元
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    # 第二層隱藏層, 包含128個神經元
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # 第三層隱藏層, 包含256個神經元
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    # 第四層為輸出層分 10 個類別
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    return model

model = build_model()
print(model.summary())

# 編譯模型
model.compile(optimizer= tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images, train_labels = preprocess(train_image, train_label)
batchsz = 128  # 設定批次大小
# 訓練模型
history = model.fit(train_images, train_labels,epochs=100,
                    batch_size = batchsz,   # 設定批次訓練大小
                    validation_split = 0.2,    # 劃分資料集的 20% 作為驗證集用
                    verbose = 2)  # 印出為精簡模式

# 繪出Loss 曲線
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# 儲存網路參數
model.save_weights('my_model_weights.h5')
print('Save Model')
del model
# 重新建立網路模型
newModel = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 網路編譯
newModel.compile(optimizer= tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 測試資料前處理
test_image, test_labels = preprocess(test_image, test_label)
# 載入網路參數
newModel.load_weights('my_model_weights.h5')
print('Load Model weight')
# 網路評估
loss, accuracy = newModel.evaluate(test_image,test_labels)
# 列印損失值與正確率
print("\n test loss : ", loss)
print("\n test accuracy : ", accuracy)