import numpy as np
import cv2
import os
import time
from skimage.transform import resize
from matplotlib import pyplot as plt
from keras.models import load_model
import InstanceNormalization

gen1 = load_model("trained_model_path")
count=0
vidcap = cv2.VideoCapture('')
success,img = vidcap.read()
print(success)
img = resize(img, (1080, 1920 ,3))

out = cv2.VideoWriter(os.path.join('results\\', '1080p.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))
vidcap2 = cv2.VideoCapture('720p.mkv')
vidcap2.read()
success,img2 = vidcap2.read()
img2 = resize(img2, (720, 1280, 3))
img1 = gen1.predict([[img],[img2]])
gen_time = 0
img_array = []
for i in range(500):
    success,img2 = vidcap2.read()
    img2 = resize(img2, (720, 1280, 3))
    start_time = time.time()
    img1 = gen1.predict([[img1[0]],[img2]])
    gen_time+= time.time() - start_time
    count+=1
    img5 = np.array(img1[0], dtype=np.uint8)
    out.write(img5)
out.release()
print("--- %s seconds ---" % gen_time)
