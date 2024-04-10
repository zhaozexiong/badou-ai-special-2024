import matplotlib.pyplot as plt

plt.figure(1)  # the first figure
plt.subplot(211)  # the first subplot in the first figure
plt.plot([1, 2, 3])  # 传入单个列表时，列表的索引值 是x轴 是y轴  （0，1）（1，2）（2，3）
plt.subplot(212)  # the second subplot in the first figure
plt.plot([4, 5, 6])

plt.subplots_adjust(hspace=0.5)

plt.figure(2)  # a second figure
plt.plot([4, 5, 6])  # creates a subplot(111) by default

plt.figure(1)  # figure 1 current; subplot(212) still current
plt.subplot(212)  # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3')  # subplot 211 title
plt.show()
plt.close()
