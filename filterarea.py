# -*- coding:utf-8 -*-

import numpy as np
import cv2

def showlabels(areacnt, labels, stats, centroids):
    font = cv2.FONT_HERSHEY_SIMPLEX
    newimg = np.zeros((500, 500, 3), dtype=np.uint8)
    for i in range(len(areacnt)):
#        newimg = np.zeros((500, 500, 3), dtype=np.uint8)
#        print(np.where(labels==areacnt[i]))
        newimg[np.where(labels==areacnt[i])] = (255, 255, 255)
        cv2.circle(newimg, (int(centroids[i][0]), int(centroids[i][1])), 2, (0, 0, 255), -1)
        area = stats[i][4]
        cv2.putText(newimg, str(area), (int(round(centroids[i][0])), int(round(centroids[i][1]))),\
                            font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('showsingle', newimg)
        cv2.waitKey(0)

def showalllabels(labels, name):
    tmpimg = np.zeros(labels.shape, dtype=np.uint8)
    tmpimg[np.where(labels>0)] = 255
    cv2.imshow(name, tmpimg)  #显示的图像是要进行笔画宽度判断的所有对象
    cv2.waitKey(0)

def getValidArea(areacnt, labels, stats, centroids):
    '''
    1. 对连通区域的长宽比、宽度、面积进行判断，舍去不符合要求的区域，如非常小或非常大的区域
    2. 连通域合并的原则：两个连通域质心的X轴差值，Y轴差值同时小于预设值时，认为这两个连通域可以合并，形成有效汉字区域
    3. 对合并后的汉字区域进行长宽比、宽度、面积的判断，舍去不符合要求的汉字区域
    4. 一块区域的边界框包含不超过两块区域，以消除文本外围包围线之类
    '''
    #第一步判断需要的常量
    valid_min_kgb = 0.2  #最小长宽比
    valid_max_kgb = 5 #最大长宽比
    valid_min_width = 8 #最小宽度 (有效的区域如汉字中的一个点，其宽度是10，这里取8)
    valid_max_width = 90 #最大宽度 (有效的区域如汉字最多宽90，大部分在80之内)
    valid_min_area = 30 #最小面积
    valid_max_area = 3200 #最大面积
    #第二步判断需要的常量
    valid_len =40  #X,Y向外扩展多少个像素，加上自己一共是n+1个卷积核（汉字大部分应该在90像素之内，如果有2个部分组成，那么平均2个部分的质心应该差距在45像素内）
    #第三步判断需要的常量
    valid_hz_min_kgb = 0.5  #最小长宽比
    valid_hz_max_kgb = 2 #最大长宽比
    valid_hz_min_width = 30 #最小宽度
    valid_hz_max_width = 70 #最大宽度
    valid_hz_min_area = 400 #最小面积(汉字区域的总面积大概在2000~4000之间，字体面积的最小区域取20%，最大区域取80%，2000*0.2=400；4000*0.8=3600)
    valid_hz_max_area = 3200 #最大面积
    #第四步判断需要的常量
    valid_includ_areas = 2 #最大可包含的区域数
    #暂时用不到的常量
    valid_min_stroke_rate = 0.2 #区域内笔画最小密度
    valid_max_stroke_rate = 0.5 #区域内笔画最大密度
    
    
    def get_stats(img, val):
        row, col = np.where(img==val)
        WIDTH = max(col)-min(col)+1
        HEIGHT = max(row)-min(row)+1
        LEFT = min(col)
        TOP = min(row)
        PIXELS = len(row)
        centerX = LEFT + round(WIDTH/2)
        centerY = TOP + round(HEIGHT/2)
        return [WIDTH, HEIGHT, LEFT, TOP, PIXELS], [centerX, centerY]
    
    print('area cnt nums:', areacnt)
    # 第1步 去除无效的区域
#    print(np.unique(labels[np.where(labels>=0)]))
#    print(areacnt)
    areacnt1=[]
    labels1 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
    stats1 = []
    centroids1 = []
    tmpimg = np.zeros(labels.shape, dtype=np.uint8)
    for n in range(1, areacnt):
        width = stats[n,cv2.CC_STAT_WIDTH]
        height = stats[n, cv2.CC_STAT_HEIGHT]
        kgb = width/height #宽高比
        area = stats[n, cv2.CC_STAT_AREA]  #面积
        '''
        kgb > valid_min_kgb and kgb < valid_max_kgb 宽高比要在0.1~10之间
        area > valid_min_area and area < valid_max_area 面积要在valid_min_area 到 valid_max_area之间
        width > valid_min_width and width < valid_max_width 区域的宽度要在valid_min_width 到 valid_max_width之间
        '''
        if kgb > valid_min_kgb and kgb < valid_max_kgb and area > valid_min_area and area < valid_max_area \
        and width > valid_min_width and width < valid_max_width: 
            areacnt1.append(n)
            labels1[np.where(labels==n)] = n
            stats1.append(stats[n])
            centroids1.append(centroids[n])
#            print('ok')
        #------------------------------------
#        print('宽高比:{0} need({1}~{2}); 面积:{3} need({4}~{5}); 宽度{6} need({7}~{8}); 密度:{9}'.format( \
#            kgb,valid_min_kgb,  valid_max_kgb, \
#            area, valid_min_area, valid_max_area, \
#            width, valid_min_width, valid_max_width, \
#            area/(width*height)))
#            
#        tmpimg[np.where(labels==n)] = 255
#        cv2.imshow('kgb & area img', tmpimg)
#        cv2.waitKey(0)
        #------------------------------------
    print('first area nums:', len(areacnt1))
#    showlabels(areacnt1, labels1, stats1, centroids1)
#    showalllabels(labels1, 'first step')
    #-------------------------------
#    print('areacnt1', areacnt1)
#    #显示所有区域和它的质心
#    tmpimg = np.zeros(labels.shape, dtype=np.uint8)
#    for i in range(len(areacnt1)):
#        tmpimg[np.where(labels1==areacnt1[i])] = 255
#        cv2.circle(tmpimg, (int(centroids1[i][0]), int(centroids1[i][1])), 2, (127, 127, 127), -1)
#        cv2.imshow('tmpimg', tmpimg)
#        cv2.waitKey(0)
#    exit()
    #areacnt1 [2, 3, 5, 6, 9, 16, 18, 32, 33, 35, 39, 41, 52, 53, 60, 75, 77, 82, 94, 100, 108, 109, 110, 111, 113, 125, 127, 131, 134, 141, 147, 148, 151, 154, 159, 178, 188, 205, 206, 208, 212, 213, 214, 216]
    #-------------------------------
    
    # 第2步 连通接近的区域
    handled_mark = []  #已经处理的区域序号
#    all_mark = [i for i in areacnt1]
    needhandle_mask = [x for x in areacnt1 if x not in handled_mark]  #待处理的区域序号
#    print('needhandle_mask', needhandle_mask)
#    print('areacnt1', areacnt1)
#    print('centroids1', centroids1)
    while len(needhandle_mask) >0:
        #得到第一个待处理区域的左上和右下坐标
        idx = areacnt1.index(needhandle_mask[0])
        left_centerX = centroids1[idx][0] -valid_len
        top_centerY = centroids1[idx][1] -valid_len
        right_centerX = centroids1[idx][0] +valid_len
        bottom_centerY = centroids1[idx][1] +valid_len
        handled_mark.append(needhandle_mask[0])
        #----------------------
#        print('开始判断一个新对象是否包含其他区域的质心....')
#        tmpimg = np.zeros((500, 500, 3), dtype=np.uint8)
#        tmpimg[np.where(labels1==needhandle_mask[0])] = (127, 127, 127)
#        cv2.rectangle(tmpimg, (int(left_centerX), int(top_centerY)), (int(right_centerX), int(bottom_centerY)),(255, 255, 255), 1)
#        cv2.imshow('tmpimg', tmpimg)
#        cv2.waitKey(0)
        #----------------------
        #判断哪些区域在卷积核里面
        for i in range(1, len(needhandle_mask)):  #从待处理列表的第二个元素开始循环
            idx1 = areacnt1.index(needhandle_mask[i])
            if centroids1[idx1][0] >= left_centerX and centroids1[idx1][0] <= right_centerX and \
                centroids1[idx1][1] >= top_centerY and centroids1[idx1][1] <= bottom_centerY:
                handled_mark.append(needhandle_mask[i])
                #--------------------------
#                tmpimg[np.where(labels1==needhandle_mask[i])] = (127, 127, 127)
#                cv2.circle(tmpimg, (int(centroids1[idx1][0]), int(centroids1[idx1][1])), 3, (255,0,0), 1)
#                cv2.imshow('tmpimg', tmpimg)
#                cv2.waitKey(0)
                #---------------------------
        handled_mark.append(-1)  #可以合并的区域之间的间隔标志
        needhandle_mask = [x for x in areacnt1 if x not in handled_mark]  #待处理的区域序号
#    print(handled_mark)
    #判断完毕，开始合并
    merge_img = np.zeros(labels.shape, dtype=np.uint8)
    firstval_from_area = 0
    areacnt2=[]
    labels2 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
    stats2 = []
    centroids2 = []
    for i in handled_mark:
        if i != -1:
            if firstval_from_area == 0:
                firstval_from_area =  i #记录需要合并区域的第一个区域的序号
                areacnt2.append(firstval_from_area)
            merge_img[np.where(labels1==i)] = 255
            labels2[np.where(labels1==i)] = firstval_from_area
            #-----------------
#            print('area number:', i)
#            cv2.imshow('mergeimg', merge_img)
#            cv2.waitKey(0)
            #-----------------
        else:
            #-------------------------
#            print('firstval_from_area', firstval_from_area)
#            view_labels = np.zeros(labels.shape, dtype=np.uint8)
#            view_labels[np.where(labels2>0)] = 255
#            cv2.imshow('view_labels', view_labels)
#            cv2.waitKey(0)
            #-------------------------
            img_stats,centerpt = get_stats(merge_img, 255)
            stats2.append(img_stats)
            centroids2.append(centerpt)
            merge_img = np.zeros(labels.shape, dtype=np.uint8)
            firstval_from_area = 0
    print('first area nums:', len(areacnt2))
#    showlabels(areacnt2, labels2, stats2,centroids2)
#    showalllabels(labels2, 'second step')

    # 第3步 对合并后的汉字区域进行过滤，筛除无效的汉字区域
    areacnt3=[]
    labels3 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
    stats3 = []
    centroids3 = []
    tmpimg = np.zeros(labels.shape, dtype=np.uint8)
    for n in range(len(areacnt2)):
        width = stats2[n][0]
        height = stats2[n][1]
        kgb = width/height #宽高比
        area = stats2[n][4]  #面积
        '''
        kgb > valid_hz_min_kgb and kgb < valid_hz_max_kgb 宽高比要在0.1~10之间
        area > valid_hz_min_area and area < valid_hz_max_area 面积要在valid_hz_min_area 到 valid_hz_max_area之间
        width > valid_hz_min_width and width < valid_hz_max_width 区域的宽度要在valid_hz_min_width 到 valid_hz_max_width之间
        '''
        if kgb > valid_hz_min_kgb and kgb < valid_hz_max_kgb and area > valid_hz_min_area and area < valid_hz_max_area \
        and width > valid_hz_min_width and width < valid_hz_max_width: 
            areacnt3.append(areacnt2[n])
            labels3[np.where(labels2==areacnt2[n])] = areacnt2[n]
            stats3.append(stats2[n])
            centroids3.append(centroids2[n])
#            print('ok')
        #------------------------------------
#        print('宽高比:{0} need({1}~{2}); 面积:{3} need({4}~{5}); 宽度{6} need({7}~{8}); 密度:{9}'.format( \
#            kgb,valid_hz_min_kgb,  valid_hz_max_kgb, \
#            area, valid_hz_min_area, valid_hz_max_area, \
#            width, valid_hz_min_width, valid_hz_max_width, \
#            area/(width*height)))
#            
#        tmpimg[np.where(labels2==areacnt2[n])] = 255
#        cv2.imshow('kgb & area img', tmpimg)
#        cv2.waitKey(0)
        #------------------------------------
    print('second area nums:', len(areacnt3))
#    showlabels(areacnt3, labels3, stats3, centroids3)
#    showalllabels(labels3, 'third step')
    
    #第4步 将包含多个汉字区域的区域进行筛除（边框）
    areacnt4=[]
    labels4 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
    stats4 = []
    centroids4 = []
    for m in range(len(areacnt3)):
        leftX = stats3[m][2]
        topY = stats3[m][3]
        rightX = leftX+stats3[m][0]
        bottomY = topY+stats3[m][1]
        include_areas = 0  #一个区域包含的子区域数
        tmpareacnt=[x for x in areacnt3 if x != areacnt3[m]]
        #-------------------------------
#        testimg = np.zeros((500, 500, 3), dtype=np.uint8)
#        testimg[np.where(labels3==areacnt3[m])] = (127, 127, 127)
#        testimg = cv2.rectangle(testimg,(leftX,topY),(rightX,bottomY),(255,255,255),1)
        #----------------------------------
        for n in range(len(tmpareacnt)):
            if leftX <=centroids3[n][0] and rightX >= centroids3[n][0] and \
                topY <= centroids3[n][1] and bottomY >= centroids3[n][1]:
                include_areas = include_areas+1
                #----------------------
#                print(leftX, topY, rightX, bottomY, '|', centroids3[n][0], centroids3[n][1])
#                cv2.circle(testimg, (int(centroids3[n][0]), int(centroids3[n][1])), 3, (255,0,0), -1)
#                cv2.imshow('testimg', testimg)
#                cv2.waitKey(0)
                #------------------------
        if include_areas <= valid_includ_areas:
#            print('include_areas', include_areas)
            areacnt4.append(areacnt3[m])
            labels4[np.where(labels3==areacnt3[m])] = areacnt3[m]
            stats4.append(stats3[m])
            centroids4.append(centroids3[m])
    print('forth area nums:', len(areacnt4))
#    showlabels(areacnt4, labels4, stats4, centroids4)
#    showalllabels(labels4, 'forth step')
    
    return areacnt4, labels4, stats4, centroids4


#def getValidArea(areacnt, labels, stats, centroids, swtimg_copy):
#    '''
#    1. 方差大于连通域的一半认为该连通域不是有效的
#    2. 连通域合并的原则：两个连通域质心的X轴差值，Y轴差值同时小于预设值时，认为这两个连通域可以合并
#    3. 连通区域的长宽比在0.1到10之间,并且面积在5到1500之间
#    4. 一块区域的边界框包含不超过两块区域，以消除文本外围包围线之类
#    '''
##    showlabels1(areacnt, labels, stats, centroids)
#    valid_len = 10  #X,Y向外扩展多少个像素，加上自己一共是n+1个卷积核
#    valid_min_kgb = 0.5  #最小长宽比
#    valid_max_kgb = 2 #最大长宽比
#    valid_min_width = 10 #最小宽度
#    valid_max_width = 300 #最大宽度
#    valid_min_area = 50 #最小面积
#    valid_max_area = 1600 #最大面积
#    valid_includ_areas = 2 #最大可包含的区域数
#    valid_min_stroke_rate = 0.2 #区域内笔画最小密度
#    valid_max_stroke_rate = 0.5 #区域内笔画最大密度
#    
#    
#    def get_stats(img, val):
##        outputfile(img, 'newimg.txt', 2)
#        row, col = np.where(img==val)
#        WIDTH = max(col)-min(col)+1
#        HEIGHT = max(row)-min(row)+1
#        LEFT = min(col)
#        TOP = min(row)
#        PIXELS = len(row)
#        centerX = LEFT + round(WIDTH/2)
#        centerY = TOP + round(HEIGHT/2)
#        return [WIDTH, HEIGHT, LEFT, TOP, PIXELS], [centerX, centerY]
#
#    #第1步
#    areacnt1=[]
#    labels1 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
#    stats1 = []
#    centroids1 = []
#    #---------------------------
##    testimg1 = np.zeros(labels.shape, dtype=np.uint8)
#    #---------------------------
#    for m in range(1, areacnt):
#        area = swtimg_copy[np.where(labels==m)]
#        mean , stddv = cv2.meanStdDev(area)  # 求均值和均方差或者用np.std(area)
#        connectarea_half = stats[m, cv2.CC_STAT_AREA]/2
#        #-------------------------------
##        print('方差：', stddv, ' 均值一半：', mean/2, ' 连通区域一半：', connectarea_half)
##        testimg = np.zeros(labels.shape, dtype=np.uint8)
##        testimg[np.where(labels==m)] = 255
##        cv2.imshow('testimg', testimg)
##        cv2.waitKey(0)
#        #----------------------------------
#        if stddv < connectarea_half:
#            areacnt1.append(m)
#            labels1[np.where(labels==m)] = m
#            stats1.append(stats[m])
#            centroids1.append(centroids[m])
#            #---------------------
##            testimg1[np.where(labels==m)] = 255
##            cv2.imshow('testimg1', testimg1)
##            cv2.waitKey(0)
#            #----------------------
#    print('area nums:', len(areacnt1))
##    showlabels(areacnt1, labels1, stats1, centroids1)
#    showalllabels(labels1)
#    
#    #第2步
#    handled_mark = []  #已经处理的区域序号
#    all_mark = areacnt1.copy()  #所有区域的序号
#    needhandle_mask = [x for x in all_mark if x not in handled_mark]  #待处理的区域序号
#    while len(needhandle_mask) >0:
#        #得到第一个待处理区域的左上和右下坐标
#        left_centerX = centroids1[0][0] -valid_len
#        top_centerY = centroids1[0][1] -valid_len
#        right_centerX = centroids1[0][0] +valid_len
#        bottom_centerY = centroids1[0][1] +valid_len
#        handled_mark.append(needhandle_mask[0])
#        #----------------------
##        tmpimg = np.zeros((500, 500, 3), dtype=np.uint8)
##        tmpimg[np.where(labels==needhandle_mask[0])] = (127, 127, 127)
##        cv2.rectangle(tmpimg, (int(left_centerX), int(top_centerY)), (int(right_centerX), int(bottom_centerY)),(255, 255, 255), 1)
##        cv2.imshow('tmpimg', tmpimg)
##        cv2.waitKey(0)
#        #----------------------
#        #判断哪些区域在卷积核里面
#        for i in range(1, len(needhandle_mask)):  #从待处理列表的第二个元素开始循环
#            if centroids1[i][0] >= left_centerX and centroids1[i][0] <= right_centerX and \
#                centroids1[i][1] >= top_centerY and centroids1[i][1] <= bottom_centerY:
#                handled_mark.append(i)
#                #--------------------------
##                print(np.where(labels==i))
##                tmpimg[np.where(labels==i)] = (0, 0, 255)
##                cv2.circle(tmpimg, (int(centroids[i, 0]), int(centroids[i, 1])), 3, (255,0,0), 1)
##                cv2.imshow('tmpimg', tmpimg)
##                cv2.waitKey(0)
#                #---------------------------
#        handled_mark.append(-1)  #可以合并的区域之间的间隔标志
#        needhandle_mask = [x for x in all_mark if x not in handled_mark]  #待处理的区域序号
##    print(handled_mark)
#    #判断完毕，开始合并
#    merge_img = np.zeros(labels.shape, dtype=np.uint8)
#    firstval_from_area = 0
#    areacnt2=[]
#    labels2 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
#    stats2 = []
#    centroids2 = []
##    color_interval = max(math.floor(255/len([x for x in handled_mark if x != -1])), 1)
#    for i in handled_mark:
#        if i != -1:
#            if firstval_from_area == 0:
#                firstval_from_area =  i #记录需要合并区域的第一个区域的序号
#                areacnt2.append(firstval_from_area)
#            merge_img[np.where(labels==i)] = 255
#            labels2[np.where(labels==i)] = firstval_from_area
#            #-----------------
##            print('area number:', i)
##            cv2.imshow('mergeimg', merge_img)
##            cv2.waitKey(0)
#            #-----------------
#        else:
#            #-------------------------
##            print('firstval_from_area', firstval_from_area)
##            view_labels1 = labels2.copy()
##            view_labels1 = cv2.convertScaleAbs(view_labels1,view_labels1);
##            cv2.imshow('view_labels1', view_labels1)
##            cv2.waitKey(0)
#            #-------------------------
#            img_stats,centerpt = get_stats(merge_img, 255)
##            print(img_stats,centerpt)
#            stats2.append(img_stats)
#            centroids2.append(centerpt)
#            merge_img = np.zeros(labels.shape, dtype=np.uint8)
#            firstval_from_area = 0
#    print('area nums:', len(areacnt2))
##    showlabels(areacnt2, labels2, stats2,centroids2)
#    showalllabels(labels2)
#    
#    #第3步
#    areacnt3=[]
#    labels3 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
#    stats3 = []
#    centroids3 = []
#    tmpimg = np.zeros(labels.shape, dtype=np.uint8)
#    for n in range(len(areacnt1)):
#        width = stats2[n][0]
#        height = stats2[n][1]
#        kgb = width/height #宽高比
#        area = stats2[n][4]  #面积
#        '''
#        kgb > valid_min_kgb and kgb < valid_max_kgb 宽高比要在0.1~10之间
#        area > valid_min_area and area < valid_max_area 面积要在valid_min_area 到 valid_max_area之间
#        width > valid_min_width and width < valid_max_width 区域的宽度要在valid_min_width 到 valid_max_width之间
#        area >= width*height*valid_min_stroke_rate and area <= width*height*valid_max_stroke_rate 意思是区域内的笔画密度要在
#          valid_min_stroke_rate 和 valid_max_stroke_rate之间
#        '''
#        if kgb > valid_min_kgb and kgb < valid_max_kgb and area > valid_min_area and area < valid_max_area \
#        and width > valid_min_width and width < valid_max_width: #\
#        #and area >= width*height*valid_min_stroke_rate and area <= width*height*valid_max_stroke_rate:
#            areacnt3.append(areacnt2[n])
#            labels3[np.where(labels2==areacnt2[n])] = areacnt2[n]
#            stats3.append(stats2[n])
#            centroids3.append(centroids2[n])
#            print('ok')
#        #------------------------------------
##        print('宽高比:{0} need(0.5~2); 面积:{1} need(50~1600); 宽度{2} need(100~300); 密度:{3}'.format(kgb, area, width, area/(width*height)))
##        tmpimg[np.where(labels2==areacnt2[n])] = 255
##        cv2.imshow('invaild kgb & area img', tmpimg)
##        cv2.waitKey(0)
#        #------------------------------------
#    print('area nums:', len(areacnt3))
##    showlabels(areacnt3, labels3, stats3, centroids3)
#    showalllabels(labels3)
#    
#    #第4步
#    areacnt4=[]
#    labels4 = np.zeros(labels.shape, dtype=np.int32)  #累积图像
#    stats4 = []
#    centroids4 = []
#    for m in range(len(areacnt3)):
#        leftX = stats3[m][2]
#        topY = stats3[m][3]
#        rightX = leftX+stats3[m][0]
#        bottomY = topY+stats3[m][1]
#        include_areas = 0  #一个区域包含的子区域数
#        tmpareacnt=[x for x in areacnt3 if x != areacnt3[m]]
#        #-------------------------------
##        testimg = np.zeros((500, 500, 3), dtype=np.uint8)
##        testimg[np.where(labels3==areacnt3[m])] = (127, 127, 127)
##        testimg = cv2.rectangle(testimg,(leftX,topY),(rightX,bottomY),(255,255,255),1)
#        #----------------------------------
#        for n in range(len(tmpareacnt)):
#            if leftX <=centroids3[n][0] and rightX >= centroids3[n][0] and \
#                topY <= centroids3[n][1] and bottomY >= centroids3[n][1]:
#                include_areas = include_areas+1
#                #----------------------
##                print(leftX, topY, rightX, bottomY, '|', centroids3[n][0], centroids3[n][1])
##                cv2.circle(testimg, (int(centroids3[n][0]), int(centroids3[n][1])), 3, (255,0,0), -1)
##                cv2.imshow('testimg', testimg)
##                cv2.waitKey(0)
#                #------------------------
#        if include_areas <= valid_includ_areas:
##            print('include_areas', include_areas)
#            areacnt4.append(areacnt3[m])
#            labels4[np.where(labels3==areacnt3[m])] = areacnt3[m]
#            stats4.append(stats3[m])
#            centroids4.append(centroids3[m])
#    print('area nums:', len(areacnt4))
##    cv2.imshow('labels4', labels4)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    showlabels(areacnt4, labels4, stats4, centroids4)
#    showalllabels(labels4)
#    return areacnt4,labels4, stats4, centroids4
