import tensorflow as tf
import matplotlib.pyplot as plt

a = tf.Variable(tf.linspace(-10.,10.,100))
#
with tf.GradientTape() as tape:
    y = tf.sigmoid(a)
da = tape.gradient(y,a)

plt.plot(a.numpy(),da.numpy()) # tensor 轉 numpy
plt.xlabel("x")
plt.ylabel("Gradient of Sigmoid(x) ")
plt.grid()
plt.show()
