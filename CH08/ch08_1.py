import tensorflow as tf
from tensorflow.keras import layers

x = tf.range(10)  # 產生10個單詞的index
# 創建 10 個單詞, 每個單詞長度為 3 的 Word Embedding層
embeddingnet = layers.Embedding(10,3)
out = embeddingnet(x)
print(out)
# 得到 embedding 層參數
print(embeddingnet.embeddings)
# 查看參數是否是可以訓練的
print(embeddingnet.embeddings.trainable)
