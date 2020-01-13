import numpy as np
import cv2
import os
import time
from skimage.transform import resize
from matplotlib import pyplot as plt

count=0
vidcap = cv2.VideoCapture('C:\\Users\\Reliance\\Desktop\\AI\\VRE\\suits_1080p.mkv')
success,img = vidcap.read()
print(success)
img = resize(img, (1080, 1920 ,3))

out = cv2.VideoWriter(os.path.join('C:\\Users\\Reliance\\Desktop\\AI\\VRE\\results\\', 'project.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))
vidcap2 = cv2.VideoCapture('C:\\Users\\Reliance\\Desktop\\AI\\VRE\\suits_720p.mkv')
vidcap2.read()
success,img2 = vidcap2.read()
img2 = resize(img2, (720, 1280, 3))
# img = cv2.imread("C:\\Users\\y.nalluri\\Desktop\\Personal\\AI\\Dataset\\hr_1080\\frame0.jpg")
# img2 = cv2.imread("C:\\Users\\y.nalluri\\Desktop\\Personal\\AI\\Dataset\\mov_720_f\\frame1.jpg")
img1 = gen1.predict([[img],[img2]])
gen_time = 0
img_array = []
for i in range(500):
#     print("frame"+str(i))
#     img1 = cv2.imread("C:\\Users\\y.nalluri\\Desktop\\Personal\\AI\\Dataset\\hr_1080\\frame"+str(i)+".jpg")
#     img3 = cv2.imread('C:\\Users\\y.nalluri\\Desktop\\Personal\\AI\\Dataset\\hr_1080\\frame%d.jpg')
#     img2 = cv2.imread("C:\\Users\\y.nalluri\\Desktop\\Personal\\AI\\Dataset\\mov_720_f\\frame"+str(i+1)+".jpg")
    success,img2 = vidcap2.read()
    img2 = resize(img2, (720, 1280, 3))
    start_time = time.time()
    img1 = gen1.predict([[img1[0]],[img2]])
    gen_time+= time.time() - start_time
    count+=1
    img5 = np.array(img1[0], dtype=np.uint8)
    out.write(img5)
#     plt.imshow(img1[0])
#     cv2.imwrite(os.path.join('C:\\Users\\Reliance\\Desktop\\AI\\VRE\\results\\3', "vgg_0.1_3_10_10_%d.jpg")% count, img1[0])
out.release()
print("--- %s seconds ---" % gen_time)
# img_2 = gen2.predict(list)
# img_3 = gen3.predict(list)
# img_4 = gen4.predict(list)
