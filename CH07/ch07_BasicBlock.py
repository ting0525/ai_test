import tensorflow as tf
from tensorflow.keras import layers,Sequential

class BasicBlock(layers.Layer):
    # 定義殘差模塊類別
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # f(x)包含了 2 個普通卷積層，創建卷積層 1 =>(3x3),64
        self.conv1 = layers.Conv2D(filter_num, (3, 3),
                                   strides=stride, padding = 'same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 創建卷積層 2  =>(3x3),64
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1,
                                   padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:  # 插入 identity 層
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1),
                                              strides=stride))
        else:  # 否則，直接連接
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # 前向計算
        out = self.conv1(inputs) # 通過第一個卷積層
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) # 通過第二個卷積層
        out = self.bn2(out)
        # inputs 通過 identity()轉換
        identity = self.downsample(inputs)
        # f(x)+x 運算
        output = layers.add([out, identity])
        # 再通過relu激勵函數計算並傳回
        output = tf.nn.relu(output)
        return output



from tensorflow import keras
# 設定 ResBlock 模塊數目內部通道數。
class ResNet(keras.Model):
    # 第一個參數 layer_dims：[2, 2, 2, 2] 共 4個 Res Block，
    # 每個包含2個Basic Block
    # 第二個參數 num_classes：設定全連接輸出個數，這邊是指輸出有多少類。
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()

        # 預處理層；可以藉由此層來設定一開始的通道數與欲輸入的特徵圖大小
        self.Prelayer = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1),
                             padding='same')
        ])

        # 創建4個Res Block
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # 通過 Pooling 層將寬與高降低為1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最後連接一個全連接層分類
        self.fc = layers.Dense(num_classes,activation = 'softmax')

    def call(self,inputs, training=None, **kwargs):
        # 完成前向運算過程
        x = self.Prelayer(inputs)
        # 前項計算通過四個 resblock 模塊
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # shape爲 [batchsize, channel]
        x = self.avgpool(x)
        # [b, 10]
        x = self.fc(x)
        return x

    # 製作多個殘差模塊的堆疊
    def build_resblock(self, filter_num, blocks, stride=1):

        Resblock = Sequential()
        # 堆疊的第一個 BasicBlock 步長不會是 1, 所以進行下採樣
        Resblock.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):  # 其他 BasicBlock 步長都為1
            # 這裏stride設置爲1，只會在第一個Basic Block做一個下采樣。
            Resblock.add(BasicBlock(filter_num, stride=1))
        return Resblock


def resnet18():
    # 通過調整內部模塊個數可以完成不同的 resnet 網路
    return ResNet([2, 2, 2, 2])

