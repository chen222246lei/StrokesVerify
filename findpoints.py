# -*- coding:utf-8 -*-

import numpy as np
import cv2, math
from matplotlib import pyplot as plt

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
     
def getValidXY(x, y, w, h):
    if y < 0:
        y = 0
    if y >= h:
        y = h-1
    if x < 0:
        x = 0
    if x >= w:
        x = w-1
    return x, y
    
def getDirectionsPts(img):
    image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    #temp = np.ones(img.shape,np.uint8)*255 
    #for i in range(len(contours)):
    #    print(contours[i])
    #    temp = cv2.drawContours(temp, contours, i, (0, 255, 0), 1)
    #    cv2.imshow('image', temp)
    #    cv2.waitKey(0)
    #exit()

    new_contours=[]
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            new_contours.append(contours[i][j])
    new_contours = np.array(new_contours, dtype=np.int32)
    box2d = cv2.minAreaRect(new_contours)
    points=cv2.boxPoints(box2d)
    points=np.array(points, dtype=np.int) 
#    cv2.polylines(img, [points], True, (255, 255, 255), 1, cv2.LINE_AA)
#    cv2.imshow('img', img)
#    cv2.waitKey(0)
    angle = box2d[2]  #计算角度，取绝对值
#    print('angle:', angle)
    if angle == 0:
        '''
               b     c 
               ┌───┐
               │     │
               └───┘
               a     d
        对象是四四方方的，倾斜角度是0
        '''
        # 第一步 计算得到 ab 线段之间的点的坐标
        ab_pts=[]
        ab_pts.append([points[0][0], points[0][1]]) 
        calcY = points[0][1]
        while calcY>points[1][1]:
            calcY = calcY -1
            ab_pts.append([points[0][0], calcY])
        ab_pts=np.array(ab_pts, dtype=np.uint)
#        for i in range(len(ab_pts)):
#            cv2.circle(img, (ab_pts[i][0], ab_pts[i][1]), 1, (200, 200, 200), -1)
        
        # 第二步 计算得到 bc 线段之间的点的坐标
        bc_pts=[]
        bc_pts.append([points[1][0], points[1][1]])
        calcX = points[1][0]
        while calcX<=points[2][0]:
            calcX = calcX +1
            bc_pts.append([calcX, points[1][1]])
        bc_pts=np.array(bc_pts, dtype=np.uint)
#        for i in range(len(bc_pts)):
#            cv2.circle(img, (bc_pts[i][0], bc_pts[i][1]), 1,  (200, 200, 200), -1)
        
        # 第三步 计算 ab 线段中所有点在 bc 方向的坐标（是一个二维列表，其中的每一个字列表都是 bc 线段方向的一系列坐标）
        bcdirection_pts=[]
        ptcnt = len(bc_pts)
        for i in range(len(ab_pts)):
            calcX = ab_pts[i][0]
            calcY = ab_pts[i][1]
            n=0
            tempts = []
            calcX, calcY = getValidXY(calcX, calcY, img.shape[1], img.shape[0])  # 1.限制坐标超出范围
            tempts.append([calcX, calcY])   # 1.限制坐标超出范围
#            tempts.append(ab_pts[i])  #2.不限制坐标范围
            while n<=ptcnt:
                calcX = calcX+1
                if calcX >= img.shape[1]:     # 1.限制坐标超出范围
                    calcX = img.shape[1]-1    # 1.限制坐标超出范围
                tempts.append([calcX, calcY])
                n = n+1
            bcdirection_pts.append(tempts)
#        print('bcdirection_pts', bcdirection_pts)
        bcdirection_pts = np.array(bcdirection_pts, dtype=np.uint)
        
        # 第四步 计算 bc 线段中所有点在 ab 方向的坐标（是一个二维列表，其中的每一个字列表都是 ab 线段方向的一系列坐标）
        abdirection_pts=[]
        ptcnt = len(ab_pts)
        for i in range(len(bc_pts)):
            calcX = bc_pts[i][0]
            calcY = bc_pts[i][1]
            n=0
            tempts = []
            calcX, calcY = getValidXY(calcX, calcY, img.shape[1], img.shape[0])    # 1.限制坐标超出范围
            tempts.append([calcX, calcY])    # 1.限制坐标超出范围
#            tempts.append(bc_pts[i])  #2.不限制坐标范围
            while n<=ptcnt:
                calcY = calcY+1
                if calcY >= img.shape[0]:    # 1.限制坐标超出范围
                    calcY = img.shape[0]-1    # 1.限制坐标超出范围
                tempts.append([calcX, calcY])
                n = n+1
            abdirection_pts.append(tempts)
#        print('abdirection_pts', abdirection_pts)
        abdirection_pts = np.array(abdirection_pts, dtype=np.uint)
    else:
        ''' 
                 c
            b ╱╲ d
               ╲╱
                a
        对象是倾斜的,顺时针的角度是正值，逆时针的角度是负值，上图中box2d[2]的夹角是-45度(从a点看是逆时针旋转了45度)
        '''
        if int(abs(angle)) == 45 or int(abs(angle)) == 135:
            '''
            如果从 a 点逆时针旋转的角度等于45或135度的对象
            斜率k=abs(angle)，通过 A 点算出 b，需要通过B点验证；再通过同一个斜率，通过 C 点算出 b，需要通过D点验证；即A-->B C-->D的斜率相同
            斜率k=angle后，通过 B 点算出 b，需要通过C点验证；再通过同一个斜率，通过 D 点算出 b，需要通过A点验证；即B-->C D-->A的斜率相同
            '''
            ab_cd_k = math.tan(math.radians(abs(angle)))  #计算得到 ab、cd 斜率
            ab_b = points[0][1]-ab_cd_k*points[0][0] #计算 ab 线段的 b 值；cd 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
            bc_da_k = math.tan(math.radians(angle))  #计算得到 bc、da 斜率
            bc_b = points[1][1]-bc_da_k*points[1][0] #计算 bc 线段的 b 值；da 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
        else:
            '''
            斜率k=90-abs(angle)，通过 A 点算出 b，需要通过B点验证；再通过同一个斜率，通过 C 点算出 b，需要通过D点验证；即A-->B C-->D的斜率相同
            斜率k=angle后，通过 B 点算出 b，需要通过C点验证；再通过同一个斜率，通过 D 点算出 b，需要通过A点验证；即B-->C D-->A的斜率相同
            '''
            ab_cd_k = math.tan(math.radians(90-abs(angle)))  #计算得到 ab、cd 斜率
            ab_b = points[0][1]-ab_cd_k*points[0][0] #计算 ab 线段的 b 值；cd 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
            bc_da_k = math.tan(math.radians(angle))  #计算得到 bc、da 斜率
            bc_b = points[1][1]-bc_da_k*points[1][0] #计算 bc 线段的 b 值；da 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
        
        # 第一步 计算得到 ab 线段之间的点的坐标
    #    print(ab_cd_k*points[1][0]+ab_b)  #计算得到 b 点坐标
        #求a,b 两点的x轴和y轴比值
        xdiff = abs(points[1][0] - points[0][0])
        ydiff = abs(points[1][1] - points[0][1])
#        print('xdiff, ydiff', xdiff, ydiff)
        ab_pts=[]
        ab_pts.append([points[0][0], points[0][1]])
        if xdiff >= ydiff:
            #接近躺着，让x轴-1，计算y轴
            calcX = points[0][0]
            while calcX>points[1][0]:
                calcX = calcX -1
                calcY = ab_cd_k*calcX+ab_b
                ab_pts.append([calcX, calcY])
        else:
            #接近竖着，让y轴-1，计算x轴
            calcY = points[0][1]
            while calcY >= points[1][1]:
                calcY = calcY-1
                calcX = (calcY-ab_b)/ab_cd_k
                ab_pts.append([calcX, calcY])
    #    print('ab_pts', ab_pts)
        ab_pts=np.array(ab_pts, dtype=np.uint)
    #    for i in range(len(ab_pts)):
    #        cv2.circle(img, (ab_pts[i][0], ab_pts[i][1]), 1,  (200, 200, 200), -1)
            
        # 第二步 计算得到 bc 线段之间的点的坐标
    #    print(bc_da_k*points[2][0]+bc_b)  #计算得到 c 点坐标
        #求b,c 两点的x轴和y轴比值
        xdiff = abs(points[2][0] - points[1][0])
        ydiff = abs(points[2][1] - points[1][1])
#        print('xdiff, ydiff', xdiff, ydiff)
        bc_pts=[]
        bc_pts.append([points[1][0], points[1][1]])
        if xdiff >= ydiff:
            #接近躺着，让x轴+1，计算y轴
            calcX = points[1][0]
            while calcX<=points[2][0]:
                calcX = calcX+1
                calcY = bc_da_k*calcX+bc_b
                bc_pts.append([calcX, calcY])
        else:
            #接近竖着，让y轴-1，计算x轴
            calcY = points[1][1]
            while calcY >= points[2][1]:
                calcY = calcY-1
                calcX = (calcY-bc_b)/bc_da_k
                bc_pts.append([calcX, calcY])
    #    print(bc_pts)
        bc_pts = np.array(bc_pts, dtype=np.uint)
    #    for i in range(len(bc_pts)):
    #        cv2.circle(img, (bc_pts[i][0], bc_pts[i][1]), 1,  (200, 200, 200), -1)

        # 第三步 计算 ab 线段中所有点在 bc 方向的坐标（是一个二维列表，其中的每一个字列表都是 bc 线段方向的一系列坐标）
        xdiff = abs(points[2][0] - points[1][0])
        ydiff = abs(points[2][1] - points[1][1])
        bcdirection_pts=[]
        ptcnt = len(bc_pts)
        if xdiff >= ydiff:
            for i in range(len(ab_pts)):
                calcX = ab_pts[i][0]
                bc_b = ab_pts[i][1]-bc_da_k*ab_pts[i][0] #计算 bc 线段的 b 值；da 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
                tempts=[]
                calcX, calcY = getValidXY(ab_pts[i][0], ab_pts[i][1], img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                tempts.append([calcX, calcY])    # 1.限制坐标超出范围
#                tempts.append(ab_pts[i])  #2.不限制坐标范围
                n = 0
                while n<=ptcnt:
                    n = n+1
                    calcX = calcX+1
                    calcY = bc_da_k*calcX+bc_b
                    calcX, calcY = getValidXY(calcX, calcY, img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                    tempts.append([calcX, calcY])  #2.不限制坐标范围
                bcdirection_pts.append(tempts)
        else:
            for i in range(len(ab_pts)):
                calcY = ab_pts[i][1]
                bc_b = ab_pts[i][1]-bc_da_k*ab_pts[i][0] #计算 bc 线段的 b 值；da 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
                tempts=[]
                calcX, calcY = getValidXY(ab_pts[i][0], ab_pts[i][1], img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                tempts.append([calcX, calcY])    # 1.限制坐标超出范围
#                tempts.append(ab_pts[i])  #2.不限制坐标范围
                n = 0
                while n<=ptcnt:
                    n = n+1
                    calcY = calcY-1
                    calcX = (calcY-bc_b)/bc_da_k
                    calcX, calcY = getValidXY(calcX, calcY, img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                    tempts.append([calcX, calcY])  #2.不限制坐标范围
                bcdirection_pts.append(tempts)
#        print('bcdirection_pts', bcdirection_pts)
        bcdirection_pts = np.array(bcdirection_pts, dtype=np.uint)
        
        # 第四步 计算 bc 线段中所有点在 ab 方向的坐标（是一个二维列表，其中的每一个字列表都是 ab 线段方向的一系列坐标）
        xdiff = abs(points[1][0] - points[0][0])
        ydiff = abs(points[1][1] - points[0][1])
        abdirection_pts=[]
        ptcnt = len(ab_pts)
        if xdiff >= ydiff:
            for i in range(len(bc_pts)):
                calcX = bc_pts[i][0]
                ab_b = bc_pts[i][1]-ab_cd_k*bc_pts[i][0] #计算 bc 线段的 b 值；da 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
                tempts=[]
                calcX, calcY = getValidXY(bc_pts[i][0], bc_pts[i][1], img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                tempts.append([calcX, calcY])    # 1.限制坐标超出范围
#                tempts.append(bc_pts[i])  #2.不限制坐标范围
                n = 0
                while n<=ptcnt:
                    n = n+1
                    calcX = calcX+1
                    calcY = ab_cd_k*calcX+ab_b
                    calcX, calcY = getValidXY(calcX, calcY, img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                    tempts.append([calcX, calcY])  #2.不限制坐标范围
                abdirection_pts.append(tempts)
        else:
            for i in range(len(bc_pts)):
                calcY = bc_pts[i][1]
                ab_b = bc_pts[i][1]-ab_cd_k*bc_pts[i][0] #计算 bc 线段的 b 值；da 线段的 b 值需要重新计算，本算法中用不到，所以无需计算
                tempts=[]
                calcX, calcY = getValidXY(bc_pts[i][0], bc_pts[i][1], img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                tempts.append([calcX, calcY])    # 1.限制坐标超出范围
#                tempts.append(bc_pts[i])  #2.不限制坐标范围
                n = 0
                while n<=ptcnt:
                    n = n+1
                    calcY = calcY+1
                    calcX = (calcY-ab_b)/ab_cd_k
                    calcX, calcY = getValidXY(calcX, calcY, img.shape[1], img.shape[0])    # 1.限制坐标超出范围
                    tempts.append([calcX, calcY])  #2.不限制坐标范围
                abdirection_pts.append(tempts)  
#        print('abdirection_pts', abdirection_pts)
        abdirection_pts = np.array(abdirection_pts, dtype=np.uint)
    
    return abdirection_pts, bcdirection_pts

if __name__ == '__main__':
    img = cv2.imread('test.jpg', 0)
    ret, img= cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    abdirection_pts, bcdirection_pts = getDirectionsPts(img)
    n = 0
    for linepts in bcdirection_pts:
        n = n+1
        for pt in linepts:
            cv2.circle(img, (pt[0], pt[1]), 1,  (127, 127, 127), -1)
        if n == 3:
            break
    n = 0
    for linepts in abdirection_pts:
        n = n+1
        for pt in linepts:
            cv2.circle(img, (pt[0], pt[1]), 1,  (127, 127, 127), -1)
        if n == 3:
            break
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    plt.subplot(111), plt.imshow(img), plt.title('original'), plt.xticks([]), plt.yticks([])
    plt.show()
