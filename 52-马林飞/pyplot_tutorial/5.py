import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0., 10.)

lines = plt.plot(x, x ** 2, 'r-', x / 2, x * 3, 'g*')
plt.setp(lines[0], linewidth=3.0, color='y')  # 用来设置线段的属性

plt.show()
