import tensorflow as tf

arr = tf.range(10)   # 創建 0~10向量
print(arr[10:0:-1])  # 逆向存取 9~1 的元素
print(arr[::-1])   # 逆向全部存取
print(arr[::-2])   # 逆向間隔採樣 2 存取