'''
该文件用于对beauty score进行数据预处理
'''
import xlwt#write
import xlrd#read
import xlutils#edit
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

##########  
"""
   Rater   Filename  Rating  original Rating
0      1    CF1.jpg       3              NaN
1      1   CF10.jpg       3              NaN
2      1  CF100.jpg       1              NaN
3      1  CF101.jpg       2              NaN
4      1  CF102.jpg       3              NaN
"""
ratings = pd.read_excel('CM_Ratings.xlsx')

filenames = ratings.groupby('Filename').size().index.tolist() #得到所有的文件名

labels = []

image_score_mean__dict = {}

for filename in filenames:
    df = ratings[ratings['Filename'] == filename] #找到当前文件的所有行
    # count = Counter(df['Rating']).most_common(1)[0][0] #对当前文件评分个数最多的一个分数
    score_mean = round(df['Rating'].mean(), 2) #求出当前文件的平均得分，保留两位小数
    image_score_mean__dict[filename] = score_mean
    # labels.append({'Filename': filename, 'count': count, 'score_mean': score_mean})

print(image_score_mean__dict)
sys.exit(1)
# labels_df = pd.DataFrame(labels)
# print(labels_df)

###############

df = pd.DataFrame(pd.read_excel('CM_Ratings.xlsx'))
df1 = df[['Filename', 'Rating']]
# print(df1)

# Asian Female分数出现的次数统计，以下两个列表一一对应
AF_frequency_list = []
AF_ratings_list = []

ratings_dict = {}

for index, row in df1.iterrows():
    filename = str(df1.loc[index]['Filename'])
    rating = int(df1.loc[index]['Rating'])
    ratings_dict[filename] = rating
print(ratings_dict)


########

# # AF_ratings_frequency_dict = {4: 34, 5: 12, 3: 14, 2: 7, 1: 33}
# AF_ratings_list = list(AF_ratings_frequency_dict.keys())
# AF_frequency_list = list(AF_ratings_frequency_dict.values()) 

# #做高斯拟合
# x = AF_ratings_list
# y = AF_frequency_list
# print(x, y)

# # def gaussian(x,*param):
# #     return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+\
# #            param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))
 
# # popt,pcov = curve_fit(gaussian,x,y,p0=[3,4,3,6,1,1], maxfev=500000)

# # # plt.plot(x,y,'b*:',label='data')
# # plt.plot(x,gaussian(x,*popt),'r.:',label='gaussian fitting')
# # plt.legend()
# # plt.show()

# # 一个输入序列，4个未知参数，2个分段函数
# def piecewise_linear(x, x0, y0, k1, k2):
# 	# x<x0 ⇒ lambda x: k1*x + y0 - k1*x0
# 	# x>=x0 ⇒ lambda x: k2*x + y0 - k2*x0
#     return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, 
#                                    lambda x:k2*x + y0-k2*x0])

# # 用已有的 (x, y) 去拟合 piecewise_linear 分段函数
# p , e = optimize.curve_fit(piecewise_linear, x, y)

# xd = np.linspace(0, 5, 100)
# plt.plot(x, y, "*")
# plt.plot(xd, piecewise_linear(xd, *p))
# plt.show()