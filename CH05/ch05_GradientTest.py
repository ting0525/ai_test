import tensorflow as tf

def gradient_test():
    x = tf.constant(4.0,tf.float32)

    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
        tape.watch(x)  # watch 可以確保 x 被 tape 追蹤
        y1 = 3*x
        y2 = 3*x*x
    dy1_dx = tape.gradient(target=y1,sources=x)  # 求 y1 對 x 變量的梯度(導數)
    dy2_dx = tape.gradient(target=y2, sources=x)  # 求 y2 對 x 變量的梯度(導數)
    print("dy1_dx:",dy1_dx)
    print("dy2_dx:", dy2_dx)

if __name__=="__main__":
    gradient_test()
