import tensorflow as tf

pic = tf.ones([3,4])
# 第一維度 : 上填充兩排 0,下填充四排 0
# 第二維度 : 左填充三排 0,右填充一排 0
newpic = tf.pad(pic,[[2,4],[3,1]])
print(newpic)