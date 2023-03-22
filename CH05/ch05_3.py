import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-10.,10.,100)
y = tf.nn.relu(x)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("Relu(x)")
plt.title("Relu Function")
plt.grid()
plt.show()

