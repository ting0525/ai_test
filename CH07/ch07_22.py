import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,\
    GlobalMaxPool2D,Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

# 將資料做一個歸一化的動作
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x,[28,28,1])
    # 將資料大小轉為 224x224
    X = tf.image.resize(x, (224, 224))
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# 設定批次大小
batchs = 50
# 載入mnist 資料集 60000張訓練資料 , 10000張測試資料, 每張大小為 28x28
(train_Data, train_Label), (test_Data, test_Label) = mnist.load_data()
# 將訓練集資料打散
db = tf.data.Dataset.from_tensor_slices((train_Data, train_Label))
db = db.map(preprocess).shuffle(10000).batch(batchs)

db_test = tf.data.Dataset.from_tensor_slices((test_Data, test_Label))
db_test = db_test.map(preprocess).batch(batchs)

model = Sequential([
    # 第一塊 nin block
    Conv2D(192,kernel_size=5,strides=1, padding='SAME', activation='relu'
           ,input_shape=(28,28,1)),
    # 加入2個 1X1 的卷積 (NiN主要使用1×1卷積層來替代全連線層)
    Conv2D(160,kernel_size=1,strides=1, padding= 'VALID', activation='relu'),
    Conv2D(96,kernel_size=1,strides=1, padding= 'VALID', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), padding= 'SAME', strides=2),
    # 第二塊 nin block
    Conv2D(192, kernel_size=5, strides=1, padding='SAME', activation='relu'),
    # 加入2個 1X1 的卷積 (NiN主要使用1×1卷積層來替代全連線層)
    Conv2D(192, kernel_size=1, strides=1, padding='VALID', activation='relu'),
    Conv2D(192, kernel_size=1, strides=1, padding='VALID', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), padding= 'SAME', strides=2),
    # 第三塊 nin block
    Conv2D(192, kernel_size=5, strides=1, padding='SAME', activation='relu'),
    # 加入2個 1X1 的卷積 (NiN主要使用1×1卷積層來替代全連線層)
    Conv2D(192, kernel_size=1, strides=1, padding='VALID', activation='relu'),
    Conv2D(10, kernel_size=1, strides=1, padding='VALID', activation='relu'),
    # 最後用最大平均池化
    GlobalMaxPool2D(),
    Softmax(-1)
])

print(model.summary())


# 編譯與訓練網路
model.compile(optimizer= 'adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
History = model.fit(db,epochs=20,validation_data=db_test,validation_freq=1)

val_acc = History.history['val_accuracy']
acc = History.history['accuracy']
val_loss = History.history['val_loss']
loss = History.history['loss']