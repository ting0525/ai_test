from tensorflow.keras import layers
hidden_dim = 3  # 隱藏層維度
SimpleRNNCell = layers.SimpleRNNCell(hidden_dim)
SimpleRNNCell.build(input_shape=(2,4,5))
# 查看 SimpleRNNCell 內部可訓練的參數
print(SimpleRNNCell.trainable_variables)
