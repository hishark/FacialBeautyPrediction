'''
该文件用于对beauty score进行数据预处理
'''
import xlwt#write
import xlrd#read
import xlutils#edit
import pandas as pd#用pandas操作excel
import sys
import time

# start = time.process_time()
AF_ratings = pd.read_excel('test_Ratings.xlsx')
# end = time.process_time()
# print('read time:', end - start)

filenames = AF_ratings.groupby('Filename').size().index.tolist() #得到excel文件里的[Filename]列的所有数据

labels = [] #{文件名:分数}的列表

score_mean_dict = {} #整个字典存每个图片的平均分 {image: score}

for filename in filenames:
    df = AF_ratings[AF_ratings['Filename'] == filename]
    print('df>>>',df)
    score = round(df['Rating'].mean(), 2) #round(x)返回浮点数x的四舍五入值
    labels.append({'Filename': filename, 'score': score})
    score_mean_dict[filename] = score

labels_df = pd.DataFrame(labels)

'''
        Filename  score
0        AF1.jpg   2.33
1       AF10.jpg   3.43
2      AF100.jpg   2.90
3     AF1000.jpg   3.97
4     AF1001.jpg   3.73
...          ...    ...
1995   AF995.jpg   3.02
1996   AF996.jpg   3.37
1997   AF997.jpg   3.73
1998   AF998.jpg   3.35
1999   AF999.jpg   3.50
[2000 rows x 2 columns]
'''
print(labels_df, type(labels_df))