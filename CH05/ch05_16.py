from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 加載，預處理數據集
dataset = np.loadtxt("pima_indians_diabetes.csv", delimiter=",")
data = dataset[:, 0:8]   # 資料集
label = dataset[:, 8]     # 標籤

print("data.shape : ", data.shape)   # 印出資料集的維度
print("label.shape : ",label.shape)  # 印出標籤維度

# 1. 定義模型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())  # 印出網路資訊

# 2. 編譯模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3. 訓練模型   迭代150次、批處理大小為10,
history = model.fit(data, label, epochs=100, batch_size=10,
                    validation_split = 0.2,    # 劃分資料集的 20% 作為驗證集用
                    verbose = 2)  # 印出為精簡模式
print("history: ",history.history)  # 印出歷史紀錄


# 4. 評估模型
loss, accuracy = model.evaluate(data, label)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5. 數據預測
probabilities = model.predict(data)
# 將 probabilities 的輸出值透過np.round()做四捨五入
predictions = [float(np.round(x)) for x in probabilities]
# 計算預測結果跟真實結果的平均差距
accuracy = np.mean(predictions == label)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))