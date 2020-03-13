# -*- coding:utf-8 -*-

import numpy as np
import cv2, sys
from os.path import realpath, dirname
from filterarea import getValidArea
from matplotlib import pyplot as plt
from strokewidth import getStrokeWidth

def outputfile(tmpimg, filename, type):
    if type==1:  #如果数据是一维
        s=''
        with open(filename, 'w') as f_obj:
            for i in range(len(tmpimg)):
                s = s+str(tmpimg[i])+','
            f_obj.write(s+'\n')
    if type==2:  #如果数据是二维
        r, c = tmpimg.shape
        with open(filename, 'w') as f_obj:
            for i in range(r):
                s=''
                for j in range(c):
                    s = s+str(tmpimg[i, j])+','
                f_obj.write(s+'\n')
    if type==3:  #如果数据是二维
        r, c, d = tmpimg.shape
        with open(filename, 'w') as f_obj:
            for i in range(r):
                s=''
                for j in range(c):
                    for k in range(d):
                        s = s+str(tmpimg[i, j])+' '
                    s = s+','
                f_obj.write(s+'\n')

def enlargediff(sobelimg):
    sobelimg_bak = sobelimg.copy()
    sobelimg_bak = np.array(sobelimg_bak, dtype=np.uint32)
    sobelimg_bak = sobelimg_bak*10
    left_right_img = np.zeros(sobelimg.shape, dtype=np.uint8)
    top_bottom_img = np.zeros(sobelimg.shape, dtype=np.uint8)
    rows, cols = sobelimg.shape
    val = 0
    #右边减去左边
    left_right_img[:, 0] = 0
    for col in range(1, cols):
        for row in range(rows):
            val = abs(int(sobelimg_bak[row, col])-int(sobelimg_bak[row, col-1]))
            if val > 255:
                left_right_img[row, col] = 255
            else:
                left_right_img[row, col] = val
    #下边减去上边
    top_bottom_img[0, :]=0
    for row in range(1, rows):
        for col in range(cols):
            val = abs(int(sobelimg_bak[row, col])-int(sobelimg_bak[row-1, col]))
            if val > 255:
                top_bottom_img[row, col] = 255
            else:
                top_bottom_img[row, col] = val
    
    enforceimg = cv2.addWeighted(left_right_img, 0.5, top_bottom_img, 0.5, 0)
    return enforceimg

path = dirname(realpath(__file__))
filename = path+'\\'+sys.argv[1]
original_img = cv2.imread(filename)
blurimg = cv2.GaussianBlur(original_img,(5,5),0) 
gray = cv2.cvtColor(blurimg, cv2.COLOR_BGR2GRAY)
# 使用双边滤波，5 邻域直径，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
blur = cv2.bilateralFilter(gray,5,75,75)
ret, otsugray = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow('otsugray', otsugray)
#cv2.waitKey(0)

#2.1 获得梯度
sobelx64=cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=1) # 水平变化求导，从“白到黑”时求导时负数，CV_64F(np.float64)会保留原值
abs_sobelx64= np.absolute(sobelx64)
sobelx_8u = np.uint8(abs_sobelx64)
sobely64=cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=1) # 垂直变化求导，从“白到黑”时求导时负数，CV_64F(np.float64)会保留原值
abs_sobely64= np.absolute(sobely64)
sobely_8u = np.uint8(abs_sobely64)
sobelimg = cv2.addWeighted(sobelx_8u,1,sobely_8u,1,0)  #图像混合，这里γ的取值为0。
#cv2.imshow('sobelimg', sobelimg)  
#cv2.waitKey(0)
#2.2 增加对比度
inforce_sobel = enlargediff(sobelimg)
#cv2.imshow('enlargediff', inforce_sobel)  
#cv2.waitKey(0)
#2.3 闭运算
kernel = np.ones((2, 2), np.uint8)
inforce_sobel = cv2.morphologyEx(inforce_sobel, cv2.MORPH_OPEN, kernel, iterations=2)
#cv2.imshow('close_inforce_sobel', inforce_sobel)  
#cv2.waitKey(0)
#2.4二值化
ret, ostu_inforce_sobel = cv2.threshold(inforce_sobel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow('ostu_inforce_sobel', ostu_inforce_sobel)  #显示的图像是要进行笔画宽度判断的所有对象
#cv2.waitKey(0)
#2.5合并
mergeimg = cv2.bitwise_and(ostu_inforce_sobel, otsugray)
cv2.imshow('merge img', mergeimg)
cv2.waitKey(0)

#得到连通域
#areacnt, labels, stats, centroids = cv2.connectedComponentsWithStats(swtimg)
areacnt, labels, stats, centroids = cv2.connectedComponentsWithStats(mergeimg)

#过滤无效对象
new_areacnt,new_labels, new_stats, new_centroids = getValidArea(areacnt, labels, stats, centroids)

#显示需要判断笔画宽度的对象图像
needjudge_img = np.zeros(new_labels.shape, dtype=np.uint8)
needjudge_img[np.where(new_labels>0)] = 255
cv2.imshow('need judge strokes img', needjudge_img)  #显示的图像是要进行笔画宽度判断的所有对象
cv2.waitKey(0)

# 5. 对所有符合条件的对象进行笔画宽度判断，过滤掉不符合条件的对象
last_areacnt=[]
lastimg = np.zeros(new_labels.shape, dtype=np.uint8)
last_stats=[]
last_centroids=[]
#print('new_areacnt', new_areacnt)
#print('new_stats', new_stats)
#print('new_centroids', new_centroids)
#print(np.unique(new_labels[np.where(new_labels>0)]))
for i in new_areacnt:
    tmpimg = np.zeros(new_labels.shape[:2], dtype=np.uint8)
    tmpimg[np.where(new_labels==i)] = 255
    ishz = getStrokeWidth(tmpimg)
#    print('ishz', ishz, '\n------------------------')
#    cv2.waitKey(0)
    if ishz:
        idx = new_areacnt.index(i)
        last_areacnt.append(i)
        lastimg[np.where(new_labels==i)] = 255  #是汉字才放到最终图像中
        last_stats.append(new_stats[idx])
        last_centroids.append(new_centroids[idx])
#    cv2.imshow('lastimg', lastimg)  #显示最终是汉字的图像
#    print('last_areacnt', last_areacnt)
#    print('last_stats', last_stats)
#    print('last_centroids', last_centroids)
#    cv2.waitKey(0)
    
#6.  获取是汉字对象的坐标在原图中显示
for box in last_stats:
    leftX = box[2]
    topY = box[3]
    rightX = leftX+box[0]
    bottomY = topY+box[1]
    hzimg = cv2.rectangle(original_img, (leftX, topY), (rightX, bottomY), (255, 255, 255), 1)
cv2.imshow('hzimg', hzimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

