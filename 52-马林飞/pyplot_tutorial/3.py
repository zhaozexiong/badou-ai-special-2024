import matplotlib.pyplot as plt
import numpy as np

f_num = np.arange(0., 5., 0.2)
print(f_num)

plt.plot(f_num, f_num * f_num, 'ro', f_num, f_num * 2, 'g--', f_num, f_num, 'b.', linewidth=2.0)  # linewidth 表示线宽
plt.axis([0, 5, 0, 25])
plt.show()
