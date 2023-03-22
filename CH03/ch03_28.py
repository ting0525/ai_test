import matplotlib.pyplot as plt
# 設定第一條線的三個點
x1 = [1,2,3]
y1 = [2,1,2]
# 設定第二條線的三個點
x2 = [1,2,3]
y2 = [1,3,1]
# 繪製兩條曲線
plt.plot(x1, y1, label='First Line')
plt.plot(x2, y2, label='Second Line')

plt.xlabel('x')
plt.ylabel('y')
plt.title('curve figure')
plt.legend()
plt.show()

