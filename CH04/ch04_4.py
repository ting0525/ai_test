import tensorflow as tf
import numpy as np

tfconv1 = tf.convert_to_tensor([[1,2],[3,4]])  # 由 list創建張量
numArr = np.array([[1,2],[3,4]])
tfconv2 = tf.convert_to_tensor(numArr)  # 由 np.array 創建陣列再轉換成張量
print(" tfconv1 : ",tfconv1)
print(" tfconv2 : ",tfconv2)