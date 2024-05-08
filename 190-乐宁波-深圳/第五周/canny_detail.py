# 0. 灰度化
# 1. 降噪：高斯滤波
# 2. 梯度计算
# 3. 非极大值抑制
# 4. 双阈值算法检测、连接边缘

import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    image = plt.imread('img.png')
    print('image', image)

    # 1.灰度化
    image = image.mean(axis=-1)  # 这行代码是什么意思
    print('image', image)

    # 2.高斯滤波
    sigma = 0.5
    dim = 5
    Gaussian_filter = np.zeros([dim, dim])  # 高斯核
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
