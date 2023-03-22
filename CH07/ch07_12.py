import tensorflow as tf
# 定義一個 feature_map
feature_map = tf.constant([
     [0.0,4.0,3.0,2.5],
     [2.0,1.0,1.5,3.0],
     [3.0,2.0,4.0,6.0],
     [2.0,6.0,2.0,6.0]])
# 印出維度
print(feature_map.shape)
# 在 dim = 0 插入一個維度
feature_map = tf.expand_dims(feature_map,0)
print(feature_map.shape)
# 在 dim = 3 插入一個維度 =>目的要讓他變成　(1,4,4,1)
feature_map = tf.expand_dims(feature_map,-1)
print(feature_map.shape)

## 定義池化層
## 池化窗口2*2，高寬方向步長都為 1
pooling = tf.nn.max_pool(input = feature_map,
                         ksize = [1,2,2,1],
                         strides = [1,1,1,1],
                         padding='VALID')
print(pooling)