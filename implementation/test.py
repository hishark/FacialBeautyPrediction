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

# x = [4,5,3,2,1]
# y = [34,12,14,7,33]
# print(x, y)

# def gaussian(x,*param):
#     return param[0]*np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.)))+\
#            param[1]*np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))
 
# popt,pcov = curve_fit(gaussian,x,y,p0=[3,4,3,6,1,1], maxfev=100000)



# # plt.plot(x,y,'b*:',label='data')
# plt.plot(x,gaussian(x,*popt),'r.:',label='gaussian fitting')
# plt.legend()
# plt.show()


# x = np.array([4, 5, 3, 2, 1], dtype=float)
# y = np.array( [34, 12, 14, 7, 33])

# x = [4, 5, 3, 2, 1]
# y = [34, 12, 14, 7, 33]

# # 一个输入序列，4个未知参数，2个分段函数
# def piecewise_linear(x, x0, y0, k1, k2):
# 	# x<x0 ⇒ lambda x: k1*x + y0 - k1*x0
# 	# x>=x0 ⇒ lambda x: k2*x + y0 - k2*x0
#     return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, 
#                                    lambda x:k2*x + y0-k2*x0])

# # 用已有的 (x, y) 去拟合 piecewise_linear 分段函数
# p , e = optimize.curve_fit(piecewise_linear, x, y)

# xd = np.linspace(0, 15, 100)
# plt.plot(x, y, "*")
# plt.plot(xd, piecewise_linear(xd, *p))
# plt.show()

def gd(x, mu=0, sigma=1):
    """根据公式，由自变量x计算因变量的值

    Argument:
        x: array
            输入数据（自变量）
        mu: float
            均值
        sigma: float
            方差
    """
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu)**2 / (2 * sigma))
    return left * right

u = 0 # 均值μ
u01 = -2
sig = math.sqrt(0.2) # 标准差δ
sig01 = math.sqrt(1)
sig02 = math.sqrt(5)
sig_u01 = math.sqrt(0.5)
x = np.linspace(u - 3*sig, u + 3*sig, 50)
x_01 = np.linspace(u - 6 * sig, u + 6 * sig, 50)
x_02 = np.linspace(u - 10 * sig, u + 10 * sig, 50)
x_u01 = np.linspace(u - 10 * sig, u + 1 * sig, 50)
y_sig = np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)
y_sig01 = np.exp(-(x_01 - u) ** 2 /(2* sig01 **2))/(math.sqrt(2*math.pi)*sig01)
y_sig02 = np.exp(-(x_02 - u) ** 2 / (2 * sig02 ** 2)) / (math.sqrt(2 * math.pi) * sig02)
y_sig_u01 = np.exp(-(x_u01 - u01) ** 2 / (2 * sig_u01 ** 2)) / (math.sqrt(2 * math.pi) * sig_u01)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.plot(x_01, y_sig01, "g-", linewidth=2)
plt.plot(x_02, y_sig02, "b-", linewidth=2)
plt.plot(x_u01, y_sig_u01, "m-", linewidth=2)
# plt.plot(x, y, 'r-', x, y, 'go', linewidth=2,markersize=8)
plt.grid(True)
plt.show()