import tensorflow as tf
# 建立一個有線性運算與 dropout 層的網路
def MakeNN(inputs, in_size, out_size,keep_prob ,activation_function=None):
    Weights = tf.Variable(tf.random.normal ([in_size, out_size]))
    # 偏置 b 的 shape 為1行out_size列
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 調用dropout功能
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        # 如果沒有設置激活函數，則直接就把當前信號原封不動地傳遞出去
        outputs = Wx_plus_b
    else:
        # 如果設置了激活函數，則會由此激活函數來對信號進行傳遞
        outputs = activation_function(Wx_plus_b)
    return outputs

X = tf.random.normal([1,784])
ModelOut = MakeNN(X,784,10,0.5,tf.nn.relu)
print(ModelOut)
