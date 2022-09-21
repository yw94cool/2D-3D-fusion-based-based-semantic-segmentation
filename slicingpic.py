# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:36:57 2018

@author: 27386
"""


import scipy.misc as misc

def make_neg(path):
    k = 0 # 图片计数用的变量
    image = misc.imread(path) # 读取图片
    print(image.shape)
    rows, cols, = image.shape[0:2] # 获得行数和列数
    r1, r2 =  [0, 400] #初始化r1, r2
    while r2 <= rows:
        c1, c2 = [0, 400] # 每循环一次，要重新给c1,c2赋值
        while c2 <= cols:
            # 截取图片
            img = image[r1 : r2 , c1 : c2 ] 
            # 把截图的图片保存到文件中
            misc.imsave('D:/DeepSEG/test1/' + 'top'+str(k) + '.jpg', img)
            k, c1, c2 = [k + 1, c1 + 400, c2 + 400]# 更新值
        r1, r2 = [r1 + 400, r2 + 400] # 更新值
    print('finish')

if __name__ == "__main__":
    make_neg('D:/DeepSEG/test/dsm_potsdam_7_11_crop05.jpg')    