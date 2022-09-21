# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:20:25 2018

@author: 27386
"""

import os
import glob
import scipy.misc as misc
from PIL import Image


file_path = 'D:/DeepSEG/test1/resultfcn/*.tif'
file_list = glob.glob(file_path)
UNIT_SIZE = 400 # the size of image 
def pinjie(images,num):
    target = Image.new('RGB', (UNIT_SIZE*5, UNIT_SIZE*5))   # result is 5*5
    leftone = 0
    lefttwo = 0
    leftthree=0
    leftfour=0
    leftfive=0
    rightone = UNIT_SIZE
    righttwo = UNIT_SIZE
    rightthree = UNIT_SIZE
    rightfour = UNIT_SIZE
    rightfive = UNIT_SIZE
    for i in range(len(images)):
        if(i%5==0):
            target.paste(images[i], (0,leftone , UNIT_SIZE,rightone ))
            leftone += UNIT_SIZE #第一行左上角右移
            rightone += UNIT_SIZE #右下角右移
        elif(i%5 == 1):
            target.paste(images[i], (UNIT_SIZE,lefttwo ,UNIT_SIZE*2 , righttwo))
            lefttwo += UNIT_SIZE #第二行左上角右移
            righttwo += UNIT_SIZE #右下角右移 
        elif(i%5 == 2):
            target.paste(images[i], (UNIT_SIZE*2, leftthree,UNIT_SIZE*3 , rightthree))
            leftthree += UNIT_SIZE #第二行左上角右移
            rightthree += UNIT_SIZE #右下角右移 
        elif(i%5 == 3):
            target.paste(images[i], ( UNIT_SIZE*3,leftfour, UNIT_SIZE*4, rightfour))
            leftfour += UNIT_SIZE #第二行左上角右移
            rightfour += UNIT_SIZE #右下角右移 
        else:
            target.paste(images[i], (UNIT_SIZE*4,leftfive ,UNIT_SIZE*5 ,rightfive ))
            leftfive += UNIT_SIZE #第二行左上角右移
            rightfive += UNIT_SIZE #右
            
        
    quality_value = 100
    target.save(path+'merge.jpg', quality = quality_value)
path = 'D:/DeepSEG/test1/resultfcn/'
num=0
images=[]
file_list = sorted(file_list, key=lambda name: int (name[48:-9]))
for f in file_list:
    images.append(Image.open(f))

print(file_list)
    
pinjie(images,num)
num +=1
images = []    
    
    