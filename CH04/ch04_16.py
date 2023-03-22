import tensorflow as tf

str = tf.constant('I like Python Language')
str1 = tf.strings.lower(str)  # 將 str 字串通通轉成小寫
print(str1)
str2 = tf.strings.upper(str)  # 將 str 字串通通轉成大寫
print(str2)
print(tf.strings.length(str2))  # 計算 str字串長度