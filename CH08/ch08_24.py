import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
# 用 pandas 載入數據及截取某一行的數據
# 分析引擎選擇 python,前面8筆資料不取
dataItem = read_csv('yahoo_stock.csv', usecols=[4],
                     engine='python', skiprows=8)
# 讀取dataItem Series 的 value
data = dataItem.values
# 將資料型態轉換成 float32
data = data.astype('float32')
print(data.shape)  # 秀出資料維度
plt.plot(data)   # 以圖表表示出
plt.show()

def GetDataAndLabel(data,TimeStep):
    trainData, trainLabel = [], []
    for i in range(len(data)-TimeStep):
        TrainDataOne = data[i:(i+TimeStep),0]
        trainData.append(TrainDataOne)
        trainLabel.append(data[i+TimeStep,0])
    return np.array(trainData), np.array(trainLabel)

from sklearn.preprocessing import MinMaxScaler
# 將數據歸一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# 將資料切割成訓練集與測試集, 分割比例為 9:1
TrainDataNum = int(len(data) * 0.9)
TestDataNum = len(data) - TrainDataNum
# 前面 0~ TrainDataNum-1 的資料為訓練集
trainData = data[0:TrainDataNum,:]
# 從 TrainDataNum 之後的資料為測試集
testData = data[TrainDataNum:len(data),:]
TimeStep = 6
traindataNew, trainLabelNew = GetDataAndLabel(trainData, TimeStep)
testdataNew, testLabelNew = GetDataAndLabel(testData, TimeStep)
print("traindataNew.shape :",traindataNew.shape)
print("trainLabelNew.shape :",trainLabelNew.shape)
print("testdataNew.shape :",testdataNew.shape)
print("testLabelNew.shape :",testLabelNew.shape)

# 將訓練資料與測試資料的維度改為 [batch_size, time_steps, input_dim]
traindataNew = np.reshape(traindataNew,
              (traindataNew.shape[0], traindataNew.shape[1], 1))
testdataNew = np.reshape(testdataNew,
              (testdataNew.shape[0], testdataNew.shape[1], 1))
print("traindataNew.shape :",traindataNew.shape)
print("testdataNew.shape :",testdataNew.shape)

from tensorflow.keras.layers import GRU, Dense
from tensorflow import keras
model = keras.Sequential()
model.add(GRU(128,input_shape=(TimeStep,1),return_sequences=True))
model.add(GRU(64,input_shape=(TimeStep,1)))
model.add(Dense(1))
print(model.summary())

# 模型建立與訓練
model.compile(loss='mean_squared_error',
              optimizer='adam',metrics=['accuracy'])
hist = model.fit(traindataNew,trainLabelNew,
                 epochs=250,batch_size=64,verbose=1)

# 繪出每個訓練周期的損失值
loss = hist.history["loss"]
epochs = range(len(loss))
plt.plot(epochs,loss,'r-',label="Training loss")
plt.title('Training Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()