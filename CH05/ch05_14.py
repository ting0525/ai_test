# 匯入Keras的mnist模組
from tensorflow.keras.datasets import mnist
(train_Data, train_Label), (test_Data, test_Label) = mnist.load_data()

# 查看 mnist 資料集大小
print('train data=', len(train_Data))
print('test data=', len(test_Data))
# 查看 mnist 資料集維度
print('train data dim=', train_Data.shape)
print('train label dim=', train_Label.shape)

import matplotlib.pyplot as plt  # 匯入matplotlib.pyplot模組

def plot_image(data):  # 輸入的是要繪製的圖象或者是陣列
    fig = plt.gcf() # 獲取當前圖形對象
    fig.set_size_inches(4,4)  # 設定圖像大小(單位:英吋)
    plt.imshow(data, cmap='binary')  # 設定顯示圖片以及顯示方式
    plt.show() # 顯示圖片

plot_image(train_Data[0])

print('train_Label[0]', train_Label[0])

import tensorflow as tf

# 訓練參數設定
learning_rate = 0.01     # 學習律
training_epoch = 1000    # 訓練次數
batch_size = 2000         # 每次訓練大小

# MNIST 資料的前置處裡
# 將原本是 28x28 的影像大小攤平成 784, 拿來當輸入特徵.
train_Data_R, test_Data_R = train_Data.reshape([-1, 784]).astype('float32')\
    ,test_Data.reshape([-1, 784]).astype('float32')
# 資料正規化
train_Data_R, test_Data_R = train_Data_R / 255., test_Data_R / 255.
# 將資料分批並且打散
train_Data_M = tf.data.Dataset.from_tensor_slices((train_Data_R, train_Label))
train_Data_M = train_Data_M.shuffle(5000).batch(batch_size)

# 最後的 Dense(10) 且 activation 用 softmax
# 代表最後 output 為 10 個 class （0~9）的機率
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# 隨機梯度下降優化器。
optimizer = tf.keras.optimizers.SGD(learning_rate)

def cross_entropy_loss(x, y):
    # 將標籤轉換為 int 64。
    y = tf.cast(y, tf.int64)
    # 選擇交叉熵當損失函數.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # 計算平均損失
    return tf.reduce_mean(loss)

# 計算準確率
def accuracy(y_pred, y_true):
    # tf.argmax(y_pred, 1) 返回 y_pred 維度為 1 的最大索引跟正確值做比較
    correct_prediction = tf.equal(tf.argmax(y_pred, 1),
                                  tf.cast(y_true, tf.int64))
    # 計算平均正確率
    return tf.reduce_mean(tf.cast(correct_prediction,
                                  tf.float32), axis=-1)


testlossArr = []   # 記錄每一個 epoch 的損失值
testaccArr = []   # 記錄每一個 epoch 的正確率
epochs = []   # 記錄每一個 epoch 值
Testloss = 0  # 記錄測試集當下 epoch 的損失值
Testacc = 0   # 記錄測試集當下 epoch 的正確率
epoch = 0
for epoch in range(training_epoch):
    for step, (batch_data, batch_label) in enumerate(train_Data_M):
        with tf.GradientTape() as tape:
            pre_data = model(batch_data)
            # Compute loss.
            loss = cross_entropy_loss(pre_data, batch_label)
            acc = accuracy(pre_data, batch_label)
            trainable_variables = model.trainable_variables
            # 計算梯度
            gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

    # 每訓練完一個 EPOCH, 就拿測試集來測試準確率
    Testprec = model(test_Data_R)
    Testloss = cross_entropy_loss(Testprec, test_Label)
    Testacc = accuracy(Testprec, test_Label)
    print("Testloss: %f, Testaccuracy: %f" % (Testloss, Testacc))
    print(epoch)
    testlossArr.append(Testloss)
    testaccArr.append(Testacc)
    epochs.append(epoch)


plt.plot(epochs,testlossArr)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()
plt.plot(epochs,testaccArr)
plt.xlabel("epoch")
plt.ylabel("Acc")
plt.grid()
plt.show()