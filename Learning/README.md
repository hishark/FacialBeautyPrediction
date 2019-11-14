# Learning
啥也不会 不知道的就查✊🏻
## TODO
- 3.1
- 3.2
- 3.4
- 4.1
- 4.2
- 5
## Paper Contents
### Abstract
be consistent to human perception

a specific supervised learning problem of classification, regression or ranking
- supervised learning 监督学习
- classification 分类
- regression 回归
- ranking 排序

FBP is a computation problem with multiple paradigms
- multiple paradigms 多范式

propose a new diverse benchmark dataset
- benchmark dataset 基准数据集

totally 5500 frontal faces

diverse properties (male/female, Asian/Caucasian, ages)

diverse labels (face landmarks, beauty scores within [1, 5], beauty score distribution) 
- face landmarks 人脸特征点

allows different computational models with different FBP paradigms, such as appearance-based/shape-based facial beauty classification/regression model for male/female of Asian/Caucasian. 
- appearance-based 基于外观的
- shape-based 基于形状的
- classification model 分类模型
- regression model 回归模型

evaluated the SCUT-FBP5500 dataset for FBP using different combinations of feature and predictor, and various deep learning methods.
- combinations of feature 特征组合
- predictor 预测器
### 1. Introduction
It has application potential in facial makeup synthesis/recommendation, content-based image retrieval, aesthetic surgery, or face beauti- fication.

It is involved with the formulation of visual representation and predictor for the abstract concept of facial beauty.
- visual representation 可视化表现形式

various data-driven models, were introduced into FBP. 

One line of the works follows the classic pattern recognition process, which constructs the FBP system using the combination of the hand-crafted features and the shallow predictors.
- hand-crafted 手工制作的
- shallow predictors 浅层预测

The related hand-crafted feature derived from visual recognition includes the geometric fea- tures, like the geometric ratios and landmark distances, and the texture features, like the Gabor/SIFT-like features. 
- geometric features 几何特征 
- geometric ratios 几何比例
- landmark distances 特征点距离
- texture features 纹理特征

The hierarchial structure of the deep learning model allows to build an end-to-end FBP system that automatically learns both the representation and the predictor of facial beauty simultaneously from the data.

Many works indicate that FBP based on deep learning is superior to the shallow predictors with hand-crafted facial feature.

We find that FBP have been formulated the recognition of facial beauty as a specific supervised learning problem of classification, regression. It indicates that FBP is intrinsically a computation problem with multiple paradigms. 

Previous databases built under specific computation constrains would limit the performance and flexibility of the computational model trained on the dataset.

Both shallow prediction model with hand-crafted feature and the state-of-the-art deep learning models were evaluated on the dataset.

Main contributions: 
- Dataset 
- Benchmark Analysis 
- Facial Beauty Prediction Evaluation
### 2. Construction of SCUT-FBP5500 Dataset
#### 2.1 Face Images Collection
faces aged from 15 to 60 with neutral expression.

four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males.
#### 2.2 Facial Beauty Scores and Facial Landmarks
All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers

 The four subset, Asian male/female and Caucasian male/female, are labeled separately

 If the correlation coefficient of the two beauty score of the same faces is less than 0.7, the volunteer would be asked to rate this face once more to decide the final score.

 To allow geometric analysis of facial beauty, 86 facial landmarks are located to the significant facial components of each images.
 - geometric analysis
 - facial landmarks

A GUI landmarks location system is developed, where the original location of the landmarks are initialized by the active shape model (ASM) trained by the SCUT-FBP dataset. Then, the detected landmarks by ASM are modified manually by volunteers to ensure the accuracy.
### 3. Benckmark Analysis of the SCUT-FBP5500
We made benchmark analysis of the beauty scores, labelers and face landmarks of the SCUT-FBP5500 with different gender and races, including Asian female (AF), Asian male (AM), Caucasian female (CF) and Caucasian male (CM).
#### 3.1 Distribution of Beauty Scores
visualize the distribution respectively

preprocess the data and filter the outliers of the beauty scores. 
- preprocess the data 数据预处理
- filter the outliers 过滤异常值

We regard the average score of all the 60 labelers as the ground-truth. If the score of specific labeler for the same face differs from the ground-truth over 2, the score is treated as outlier
- ground-truth 标准

we visualized the score distribution of the SCUT-FBP5500 using the preprocessed data for all the four subset, respectively.

Two distribution fitting schemes are used: one is Gaussian fitting, the other is piecewise fitting 
- Gaussian fitting 高斯拟合
- piecewise fitting 分段拟合

The results indicates that the beauty scores of all the four subset can be approximately fitted by a mixed distribution model with two Gaussian components.
- Gaussian components 高斯分量
#### 3.2 Standard Deviation of Beauty Scores
We calculate the standard deviation of the scores gathered from different labelers to the ground-truth
- standard deviation 标准差

illustrate the results as histogram and as box figure
- histogram 直方图
- box figure 箱形图 箱线图

the distribution of standard deviations is similar to Gaussian distribution

most of the standard deviations are within a reasonable range of [0.6, 0.7].
- reasonable range 合理范围
#### 3.3 Correlation of Male/Female Labelers
 investigate the correlation between the male and female Asian labelers for the beauty scores

 It is consistent to the psychological research that human have better facial beauty perception for the faces from the same race.
#### 3.4 PCA Analysis of Facial Geometry
We visualize the 86-points face landmarks of the dataset using principle component analysis(PCA). 
- principle component analysis(PCA) 主成分分析

Fig.6 illustrate the mean and the five first principle component of the facial geometry of Asian
- principle component 主成分

the landmarks data of Caucasian share similar distribution to Asian faces

We observe that the face shape is one of the main component influence the face geometry of beauty
- face shape 脸部形状（是脸型嘛
### 4. FBP Evaluation via Hand-Crafted Feature and Shallow Predictor
we evaluate the SCUT-FBP5500 using the hand-crafted feature with shallow predictor
- hand-crafted 手工制作的
- shallow predictors 浅层预测
#### 4.1 Geometric Feature with Shallow Predictor
We extract a 18-dimensional ratio feature vector from the faces and formulate FBP based on different regression models, such as the linear regression(LR), Gaussian regression(GR), and support vector regression(SVR).
- 18-dimensional ratio feature vector 18维比例特征向量
- regression models 回归模型
- linear regression(LR) 线性回归
- Gaussian regression(GR) 高斯回归
- support vector regression(SVR) 支持向量回归

Comparison were performed for Caucasian and Asian subsets

the performance of different model are measured using pearson correlation coefficient (PC), maximum absolute error(MAE) and root mean square error(RMSE) after 10 folds cross validation
- pearson correlation coefficient(PC) 皮尔逊相关系数
- maximum absolute error (MAE) 最大绝对误差
- root mean square error (RMSE) 均方根误差（标准误差）
- 10 folds cross validation 10次交叉验证

The results in Table can be regarded as a baseline for the geometric analysis of FBP.
- baseline 基线
- geometric analysis 几何分析
#### 4.2 Appearance Feature with Shallow Predictor
We extract 40 Gabor feature maps from every original image in five directions and eight angles.

using two different sampling schemes that extracts some component of the Gabor feature maps as following:
- Sample 86-keypoints
- 64UniSample

use PCA to reduce the extracted feature dimension before we train the predictor. 
- the extracted feature dimension 提取的特征维数

Week12进度就汇报到这里为止，剩下的放到final👋🏻

---
### 5. FBP Evaluation via Deep Predictor
We evaluate three recently proposed CNN models with different structures for FBP, including AlexNet, ResNet-18 and ResNeXt-50:
- AlexNet <= 227*227 random crop of raw image
- ResNet-18 <= 224*224 random crop of raw image
- ResNeXt-50 <= 224*224 random crop of raw image

All these CNN models are trained by initializing weights using networks pre-trained on the ImageNet dataset.

Two different experiment settings as following:
- 5-fold cross validation
- 60% for train, 40% for test

The results illustrates that the deepest CNN-based ResNeXt-50 model obtains the best performance

all the deep CNN model are superior to the shallow predictor

It indicates the effectiveness and powerfulness of the end-to-end feature learning deep model for FBP.
- end-to-end feature learning 端到端特征学习

the accuracy of all the 5-fold cross validation is slightly higher than the results of the split of 60% training and 40% testing.
- 5-fold cross validation 5次交叉验证

data augmentation techniques may further improve the performance of the deep FBP model, which merits exploring in the future.
### 6. Conclusion
就一小段总结，再次夸了一波SCUT-FBP550。
## References
[机器学习](https://zh.wikipedia.org/wiki/机器学习) 

[人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络) 

[卷积神经网络](https://zh.wikipedia.org/wiki/卷积神经网络)

[交叉验证](https://zh.wikipedia.org/wiki/交叉验证)

[过拟合](https://zh.wikipedia.org/wiki/过拟合)

[AlexNet](https://my.oschina.net/u/876354/blog/1633143)

[PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)

[Python数据可视化-箱形图](https://zhuanlan.zhihu.com/p/34720695)

[相关系数](https://baike.baidu.com/item/%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0/3109424?fr=aladdin)

[绝对误差](https://baike.baidu.com/item/%E7%BB%9D%E5%AF%B9%E8%AF%AF%E5%B7%AE)

[均方根误差](https://baike.baidu.com/item/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE)

[hand-craft](https://blog.csdn.net/qq_22194315/article/details/82704776)

[MacOS安装caffe](https://zhuanlan.zhihu.com/p/46930024)

[CNN入门讲解：什么是微调（Fine Tune）？](https://zhuanlan.zhihu.com/p/35890660)

[active shape model (ASM)](http://ice.dlut.edu.cn/lu/ASM.html)

[ASM](https://blog.csdn.net/carson2005/article/details/8194317)

[Python之Sklearn使用教程](https://blog.csdn.net/xiaoyi_eric/article/details/79952325)

[sklearn-docs](https://sklearn.apachecn.org/docs/0.21.3/50.html)

[sklearn-user-guide](https://scikit-learn.org/stable/user_guide.html)

[PCA主成分分析提取特征脸](https://blog.csdn.net/u010975589/article/details/88025494)

[PCA-1](https://blog.csdn.net/weixin_42001089/article/details/79989788)

[PCA-2](https://blog.csdn.net/liangjun_feng/article/details/78664820)

[PCA-3](https://blog.csdn.net/u010975589/article/details/88025494)

[python调用resnet模型对人脸图片进行特征提取全连接层特征向量](https://blog.csdn.net/zuqiutxy/article/details/71156593)

[数据预处理（python）](https://www.jianshu.com/p/4f3d9a34d246)

[numpy tutorial](https://www.runoob.com/numpy/numpy-tutorial.html)