import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10)
sin_line = plt.plot(x, np.sin(x), linewidth=3.0)  # 使用 sin_line, = plt.plot(x, np.sin(x) 等同于 sin_line[0]
sin_line[0].set_antialiased(False)
plt.gca().add_line(sin_line[0])
plt.show()
