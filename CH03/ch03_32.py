import matplotlib.pyplot as plt

fig , ax = plt.subplots(2, 3, sharex = 'all', sharey = 'all')
# sharex->是否共享x軸  sharey->是否共享y軸
for i in range(2):
    for j in range(3):
        # 在各個子網格中間寫上文字
        ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')
plt.show()

