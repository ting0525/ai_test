import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-10.,10.,100)
y = tf.nn.leaky_relu(x, alpha = 0.5) #alpha是一个超参数，控制负数部分的斜率，它决定了在负数区域中的线性倾斜程度
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("leaky_relu(x)")
plt.title("leaky_relu Function")
plt.grid()
plt.show()

