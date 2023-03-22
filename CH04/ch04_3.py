import tensorflow as tf

Var1 = tf.Variable([1,2,3,4,5])   # 直接指定初值
a = tf.constant([6,7,8,9,10])     # 創建張量
Var2 = tf.Variable(a)             # 轉換成 variable 類型

import numpy as np
print("Var1 的 name :",Var1.name, "是否可求導 : ",Var1.trainable)
print("Var2 的 name :",Var2.name, "是否可求導 : ",Var2.trainable)