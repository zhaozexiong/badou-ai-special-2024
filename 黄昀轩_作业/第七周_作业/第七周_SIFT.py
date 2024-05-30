# """
# SIFT 高版本 这个步骤要改写成 sift=cv2.SIFT_create()
# """

import cv2
import matplotlib.pyplot as plt


img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

sift = cv2.SIFT_create()
kp,dst = sift.detectAndCompute(img2,None) #后面的None 是掩膜的范围 一般不用 些None
cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



fig = plt.figure(figsize=(10,5))
plt.imshow(img[:,:,::-1])
plt.title("Keypoints")
plt.xticks()
plt.yticks()
plt.show()


