# -*- coding: UTF-8 -*-
from sklearn.linear_model import LinearRegression
# import xlwt#write
# import xlrd#read
# import xlutils#edit
import pandas as pd#用pandas操作excel
import sys
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sy
from scipy import asarray as ar,exp
import math
from scipy import optimize
import os
# import keras
# import caffe
import cv2


##########  

ratings = pd.read_excel('CM_Ratings.xlsx')
"""
   Rater   Filename  Rating  original Rating
0      1    CF1.jpg       3              NaN
1      1   CF10.jpg       3              NaN
2      1  CF100.jpg       1              NaN
3      1  CF101.jpg       2              NaN
4      1  CF102.jpg       3              NaN
"""
filenames = ratings.groupby('Filename').size().index.tolist() #得到所有的文件名

labels = []

for filename in filenames:
    df = ratings[ratings['Filename'] == filename] #找到当前文件的所有行
    count = Counter(df['Rating']).most_common(1)[0][0] #对当前文件评分个数最多的一个分数
    score_mean = round(df['Rating'].mean(), 2) #求出当前文件的平均得分，保留两位小数
    labels.append({'Filename': filename, 'count': count, 'score_mean': score_mean})

labels_df = pd.DataFrame(labels)

scores = sorted(labels_df.score_mean.tolist())

lv1 = [x for x in scores if x<1]
lv2 = [x for x in scores if x>=1 and x<1.5]
lv3 = [x for x in scores if x>=1.5 and x<2]
lv4 = [x for x in scores if x>=2 and x<2.5]
lv5 = [x for x in scores if x>=2.5 and x<3]
lv6 = [x for x in scores if x>=3 and x<3.5]
lv7 = [x for x in scores if x>=3.5 and x<4]
lv8 = [x for x in scores if x>=4 and x<4.5]
lv9 = [x for x in scores if x>=4.5]
plt.bar(['1','1.5','2','2.5','3','3.5','4','4.5','5'],
       [len(x) for x in [lv1,lv2,lv3,lv4,lv5,lv6,lv7,lv8,lv9]])
plt.title('Caucasian Male Score Distribution')
# plt.show()

"""
接下来把所有图片都转成numpy.ndarray(n维数组)格式，target是这张照片的评分
x_total要用12G内存 ok顶得住
"""
img_width, img_height, channels = 350, 350, 3 #图片宽高以及channels？
sample_dir = '/Users/a777/Documents/study/研一/上学期课程/人工智能基础/FaceBeautyPrediction/SCUT-FBP550-OFFICIAL/SCUT-FBP5500_v2/Images' #图片路径
nb_samples = len(os.listdir(sample_dir))
input_shape = (img_width, img_height, channels)

x_total = np.empty((nb_samples, img_width, img_height, channels), dtype=np.float32)
y_total = np.empty((nb_samples, 1), dtype=np.float32)

print(x_total, y_total)

for i, fn in enumerate(os.listdir(sample_dir)):
    print(i, fn)
    # img = load_img('%s/%s' % (sample_dir, fn))
    img = cv2.imread('%s/%s' % (sample_dir, fn))
    x = img_to_array(img).reshape(img_height, img_width, channels)
    x = x.astype('float32') / 255.
    y = labels_df[labels_df.Filename == fn].score.values
    y = y.astype('float32')
    x_total[i] = x
    y_total[i] = y

print(x_total, y_total)

# 计算误差不担心，参考cnn计算误差的方法即可
# correlation
# mse
# rmse



