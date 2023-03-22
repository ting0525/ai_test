import tensorflow as tf

# 宣告一個字串串列張量
str = tf.constant(['I','like','Python','Language','very','much'])
# 使用給定的分隔符（默認為空分隔符）, 將給定的字符串張量列表中的字符串連接成一個張量
str1 = tf.strings.join(str,separator= '-')
print(str1)