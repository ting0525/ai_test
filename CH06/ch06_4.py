from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

fig=plt.figure()
x1,y1=make_moons(n_samples=1000,noise=0.1)
plt.title('make_moons function example')
plt.scatter(x1[:,0],x1[:,1],marker='o' ,c=y1)
plt.show()