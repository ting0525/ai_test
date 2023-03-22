import tensorflow as tf

word = tf.constant([1,2,3,4])
newword = tf.pad(word,[[1,3]])   # 左邊填一個 0, 右邊填三個 0
print(newword)