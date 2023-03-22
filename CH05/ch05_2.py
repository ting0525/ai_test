import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-10.,10,100)
y = tf.nn.tanh(x)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.title("tanh Function")
plt.grid()
plt.show()
