import cv2
import os
import random

folder = '../results/'
img = cv2.imread(folder+'50.png')  # 读取保存的任意一张图片
fps = 10	 # 根据第二节介绍的方法获取视频的 fps
# size = (img.shape[1],img.shape[0])  #获取视频中图片宽高度信息
size = (960,540)
print(size)
fourcc = cv2.VideoWriter_fourcc(*"XVID") # 视频编码格式
videoWrite = cv2.VideoWriter('1.avi',fourcc,fps,size)

files = os.listdir(folder)
files.sort(key=lambda x:int(x[:-4]))
out_num = len(files)
for i in range(out_num):
    if i%2==0:
        continue
    img = cv2.imread(folder + files[i])
    img = cv2.resize(img,size)
    videoWrite.write(img)
videoWrite.release()
