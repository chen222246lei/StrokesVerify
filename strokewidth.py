# -*- coding:utf-8 -*-

import numpy as np
import cv2
from findpoints import getDirectionsPts

'''
第一种判断方法：
笔画宽度汇总结果类似如下：统计前 calcdictcnt 个的笔画宽度和，看它占据整个笔画宽度汇总值的百分比，如果超过stroke_width_rate则表示是
汉字，如：calcdictcnt = 3，则： (44+13+6)/(44+13+6+5+3+2+1+1) 如果> stroke_width_rate 则认为是汉字
3: 44
2: 13
11: 6
10: 5
4: 3
1: 2
17: 1
8: 1

第二种判断方法：
汉字的笔画宽度大多数在5~9之间。根据此规律，如果5~9的比重小于50%则不是汉字
'''
#第一种判断方法需要的常量
#stroke_width_cnt = 3
#stroke_width_rate = 0.7 #前 stroke_width_cnt 个笔画宽度汇总值占据整个闭环宽度的百分比

#第二种判断方法需要的常量
stroke_min = 5  #有效笔画宽度的最小值
stroke_max = 9 #有效笔画宽度的最大值
stroke_valid_rate = 0.25   #笔画宽度在5~9之间的和占整个笔画宽度总和的比重

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

def putinDict(dictname, stroke_width):
    if stroke_width > 0:
        if stroke_width in dictname:
            dictname[stroke_width] = dictname[stroke_width]+1
        else:
            dictname[stroke_width] = 1
    return dictname

def sortDict(dictname):
    reverseDict={}
    for k in sorted(dictname,key=dictname.__getitem__, reverse=True):
        reverseDict[k]= dictname[k]
    return reverseDict

def ifHz_One(dictname):
    prevThree_sum = 0
    all_sum = 0
    n = 0
    for i in dictname:
        n = n+1
        if n <= stroke_width_cnt:
            prevThree_sum = prevThree_sum+dictname[i]
        all_sum = all_sum+dictname[i]
    stroke_rate = round(prevThree_sum/all_sum, 1)
    print('{0}/{1}={2} ≈ {3}'.format(prevThree_sum, all_sum,prevThree_sum/all_sum,stroke_rate))
    if stroke_rate >= stroke_width_rate:
        return 1
    else:
        return 0

def ifHz_Two(dictname):
    stroke_5to9 = 0
    stroke_all = 0
    for key in dictname:
        if key >= stroke_min and key <= stroke_max:
            stroke_5to9 = stroke_5to9+dictname[key]
        stroke_all = stroke_all+dictname[key]
#    print('stroke_5to9/stroke_all:', stroke_5to9/stroke_all)
    if stroke_5to9/stroke_all >= stroke_valid_rate:
        return 1
    else:
        return 0
        
def getStrokeWidth(img):
#    kernel = np.ones((3, 3), np.uint8)
#    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
#    ret, img= cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    cv2.imshow('image', img)  # 显示要进行笔画宽度判断的对象进行预处理后的对象
#    cv2.waitKey(0)
    abdirection_pts, bcdirection_pts = getDirectionsPts(img)
    #判断 ab 方向的笔画宽度是否满足汉字的条件
    abdirection_strokes_widths={}  #定义 ab 方向上笔画宽度的元组
    for linepts in abdirection_pts:
        stroke_width = 0
        for pt in linepts:
            if img[pt[1], pt[0]] == 255:
                stroke_width = stroke_width+1
            else:
                abdirection_strokes_widths = putinDict(abdirection_strokes_widths, stroke_width)
                stroke_width = 0
        abdirection_strokes_widths = putinDict(abdirection_strokes_widths, stroke_width)
    #第一种判断方法：
#    abdirection_strokes_widths = sortDict(abdirection_strokes_widths)  #按照笔画宽度汇总结果倒序排列，将笔画宽度相同最多的笔画排在前面
#    print('abdirection_strokes_widths', abdirection_strokes_widths)
#    abdirection_ishz = ifHz_One(abdirection_strokes_widths)
    #第二种判断方法：
    abdirection_ishz = ifHz_Two(abdirection_strokes_widths)

    #判断 bc 方向的笔画宽度是否满足汉字的条件
    bcdirection_strokes_widths={}  #定义 bc 方向上笔画宽度的元组
    for linepts in bcdirection_pts:
        stroke_width = 0
        for pt in linepts:
            if img[pt[1], pt[0]] == 255:
                stroke_width = stroke_width+1
            else:
                bcdirection_strokes_widths = putinDict(bcdirection_strokes_widths, stroke_width)
                stroke_width = 0
        bcdirection_strokes_widths = putinDict(bcdirection_strokes_widths, stroke_width)
    
    #第一种判断方法：
#    bcdirection_strokes_widths = sortDict(bcdirection_strokes_widths)  #按照笔画宽度汇总结果倒序排列，将笔画宽度相同最多的笔画排在前面
#    print('bcdirection_strokes_widths', bcdirection_strokes_widths)  
#    bcdirection_ishz = ifHz_One(bcdirection_strokes_widths)
    #第二种判断方法：
    bcdirection_ishz = ifHz_Two(bcdirection_strokes_widths)

#    #沿着 ab 方向划线，可以直观的看出是判断哪个方向的笔画的
#    n = 0
#    draw_lines = 8
#    for linepts in abdirection_pts:
#        n = n+1
#        for pt in linepts:
#            cv2.circle(img, (pt[0], pt[1]), 1, (127, 127, 127), -1)
#        if n >= draw_lines:
#            break
#    #沿着 bc 方向划线，可以直观的看出是判断哪个方向的笔画的
#    n = 0
#    for linepts in bcdirection_pts:
#        n = n+1
#        for pt in linepts:
#            cv2.circle(img, (pt[0], pt[1]), 1, (127, 127, 127), -1)
#        if n >= draw_lines:
#            break
#    cv2.imshow('img', img)
#    cv2.waitKey(0)
    
    return abdirection_ishz+bcdirection_ishz

if __name__ == '__main__':
    img = cv2.imread('test.jpg', 0)
    ishz = getStrokeWidth(img)
    print('是否汉字', ishz)
