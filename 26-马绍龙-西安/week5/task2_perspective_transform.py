import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
print(img.shape)

'''
这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[275, 275], [12, 565], [578, 357], [448, 791]])
dst = np.float32([[0, 0], [0, 450], [318, 0], [318, 450]])

# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)   #这里src中的坐标值顺序为(w,h) 或是(col,row)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(img, m, (318, 450))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
