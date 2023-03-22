import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8,8))
# 分割兩列兩行的子圖, 從左到右從上到下的第1塊。
ax_1=fig.add_subplot(2,2,1)
ax_1.text(0.3, 0.5, 'subplot(2,2,1)')
# 分割兩列兩行的子圖, 從左到右從上到下的第3塊。
ax_2=fig.add_subplot(2,2,3)
ax_2.text(0.3, 0.5, 'subplot(2,2,3)')
# 分割一列兩行的子圖, 從左到右從上到下的第2塊。
ax_3=fig.add_subplot(1,2,2)
ax_3.text(0.3, 0.5, 'subplot(1,2,2)')
fig.suptitle("multiple Subplots")
plt.show()

