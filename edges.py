import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

tag = [0 for i in range(448)]

def edge(img):
    #高斯模糊,降低噪声
    blurred = cv.GaussianBlur(img,(3,3),0)
    #灰度图像
    gray=cv.cvtColor(blurred,cv.COLOR_RGB2GRAY)
    #gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    #图像梯度
    xgrad=cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad=cv.Sobel(gray,cv.CV_16SC1,0,1)
    #计算边缘
    #50和150参数必须符合1：3或者1：2
    edge_output=cv.Canny(xgrad,ygrad,50,100)
    #cv.imwrite('G:\\edges\\'+filename+'.png',edge_output)
    #图一
    #cv.imshow("edge",edge_output)
 
    dst=cv.bitwise_and(img,img,mask=edge_output)
    #图二（彩色）
    #cv.imshow('cedge',dst)
    return dst

os.chdir('J:\\Research_lab_data_base\\train\\')

i=0

for filename in os.listdir('J:\\Research_lab_data_base\\train\\'): 
    src=cv.imread(filename)
    #cv.imshow('def',src)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    edge_output = edge(src)
    #print(str(filename).split(".")[-3])
    if(str(filename).split(".")[-3] == "fake"):
        cv.imwrite("J:\\Research_lab_data_base\\train\\"+"fake.50100"+str(i).zfill(6)+".png",edge_output)
    else:
        cv.imwrite("J:\\Research_lab_data_base\\train\\"+"real.50100"+str(i).zfill(6)+".png",edge_output)
        

    if src is None:
        continue

    i += 1
    print(i)
    #改变图片尺寸
    #img=cv.resize(src,(600,400),)
    #图三（原图）
    #cv.imshow('def',img)
    
    



#cv.waitKey(0)
#cv.destroyAllWindows()
