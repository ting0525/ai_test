import tensorflow as tf
from tensorflow.keras import layers


(train_Data, train_Label), (test_Data, test_Label) = \
    tf.keras.datasets.fashion_mnist.load_data()

train_d = train_Data.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_d = test_Data.reshape((10000, 28, 28, 1)).astype('float32') / 255

# DenseLayer，相當於每一個 dense block 中有多少個相同的 H(DenseLayer) 函數
class DenseLayer(layers.Layer):
    # growth_rate : 增長率 k
    def __init__(self, growth_rate, drop_rate):
        super(DenseLayer, self).__init__()
        # 接下來按照 bn->relu->Conv 1x1->bn->relu
        # ->Conv 3x3->Dropout(可選的,用於防止過擬合)
        self.bn1 = layers.BatchNormalization()
        # 使用 1*1 卷積核將通道數降至 4*k
        self.conv1 = layers.Conv2D(filters=4,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.bn2 = layers.BatchNormalization()
        # 使用 3*3 卷積核，使得輸出通道數為 k
        self.conv2 = layers.Conv2D(filters=growth_rate,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")
        self.dropout = layers.Dropout(rate=drop_rate)
        # 將網路存於一列表中
        self.listLayers = [self.bn1,
                           layers.Activation("relu"),
                           self.conv1,
                           self.bn2,
                           layers.Activation("relu"),
                           self.conv2,
                           self.dropout]

    def call(self, x, **kwargs):
        y = x
        for layer in self.listLayers:
            y = layer(y)
        # 每經過一個 DenseLayer，將輸入和輸出按通道拼接。
        y = layers.concatenate([x, y], axis=-1)
        return y


# 稠密塊，是由若干個相同的 DenseLayer 組成
class DenseBlock(layers.Layer):
    # num_layers 表示該 DenseBlock 存在 DenseLayer 的層數
    def __init__(self, num_Denselayers, growth_rate, drop_rate=0.5):
        super(DenseBlock, self).__init__()
        self.num_layers = num_Denselayers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.listLayers = []
        # 一個 DenseBlock 由多個 DenseLayer 構成，這邊可以將它們放入列表中。
        for _ in range(num_Denselayers):
            self.listLayers.append(DenseLayer(growth_rate=self.growth_rate,
                                              drop_rate=self.drop_rate))

    def call(self, x, **kwargs):
        for layer in self.listLayers:
            x = layer(x)
        return x

class TransitionLayer(layers.Layer):
    # out_channels 代表输出通道數 (壓縮比例由DenseNet設定)
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv2D(filters=out_channels,
                                  kernel_size=(1, 1),
                                  strides=1,
                                  padding="same")
        self.pool = layers.MaxPool2D(pool_size=(2, 2),  # 2倍下採樣
                                     strides=2,
                                     padding="same")

    def call(self, inputs, **kwargs):
        x = self.bn(inputs)
        x = tf.keras.activations.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

# DenseNet-121 整體網路結構
class DenseNet(tf.keras.Model):
    # init_featuresChannel:代表一開始的通道數，即輸入第一個稠密塊的通道數
    # growth_rate:增長率 k，指經過一個 DenseLayer 输出的特徵圖的通道數
    # block_layers:每個稠密塊中的 DenseLayer的個數
    # compression_rate:壓縮因子，其值在(0,1]範圍内
    # drop_rate：保留的機率
    def __init__(self, init_featuresChannel, growth_rate, block_layers,
                 compression_rate, drop_rate):
        super(DenseNet, self).__init__()
        # 第一層，7*7的卷積層，2倍下採樣。
        self.conv = layers.Conv2D(filters= init_featuresChannel,
                                  kernel_size=(7, 7),
                                  strides=2,
                                  padding="same")
        self.bn = layers.BatchNormalization()
        # 第二層 :最大池化層(3*3卷積和，2倍下採樣)
        self.pool = layers.MaxPool2D(pool_size=(3, 3), strides=2,
                                     padding="same")

        # 第一稠密塊:Dense Block
        # 初始通道數目(看你輸入的影像通道數)
        self.num_channels = init_featuresChannel
        self.dense_block_1 = DenseBlock(num_Denselayers=block_layers[0],
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
        # 計算第一稠密塊的輸出的通道數
        self.num_channels += growth_rate * block_layers[0]
        # 對通道數進行壓縮
        self.num_channels = compression_rate * self.num_channels
        # 第一過渡層
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))
        # 第二稠密塊:Dense Block
        self.dense_block_2 = DenseBlock(num_Denselayers=block_layers[1],
                                growth_rate=growth_rate, drop_rate=drop_rate)
        # 計算第二稠密塊的輸出的通道數
        self.num_channels += growth_rate * block_layers[1]
        # 對通道數進行壓縮
        self.num_channels = compression_rate * self.num_channels
        # 第二過渡層
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))
        # 第三稠密塊:Dense Block
        self.dense_block_3 = DenseBlock(num_Denselayers=block_layers[2],
                                growth_rate=growth_rate, drop_rate=drop_rate)
        # 計算第三稠密塊的輸出的通道數
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        # 第三過渡層
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))
        # 第四稠密塊:Dense Block
        self.dense_block_4 = DenseBlock(num_Denselayers=block_layers[3],
                                growth_rate=growth_rate, drop_rate=drop_rate)
        # 全局平均池化，輸出 size：1*1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 全連接層 (10分類)
        self.fc = layers.Dense(units=10, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.keras.activations.relu(x)
        x = self.pool(x)

        x = self.dense_block_1(x)
        x = self.transition_1(x)
        x = self.dense_block_2(x)
        x = self.transition_2(x)
        x = self.dense_block_3(x)
        x = self.transition_3(x,)
        x = self.dense_block_4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

model = DenseNet(init_featuresChannel = 1, growth_rate = 4,
                 block_layers = [6,12,24,16], compression_rate = 1,
                 drop_rate = 0)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['accuracy'])

history = model.fit(train_d, train_Label,
                    batch_size=32,
                    epochs=20,
                    verbose = 1,
                    validation_split=0.2,
                    shuffle = True)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('ACC')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
