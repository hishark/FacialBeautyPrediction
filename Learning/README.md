# Learning
啥也不会 不知道的就查✊🏻

所有PDF均来自[维基百科](https://zh.wikipedia.org/)

## Contents
### 1. Introduction
### 2. Construction of SCUT-FBP5500 Dataset
#### 2.1 Face Images Collection
#### 2.2 Facial Beauty Scores and Facial Landmarks
### 3. Benckmark Analysis of the SCUT-FBP5500
#### 3.1 Distribution of Beauty Scores
#### 3.2 Standard Deviation of Beauty Scores
#### 3.3 Correlation of Male/Female Labelers
#### 3.4 PCA Analysis of Facial Geometry
### 4. FBP Evaluation via Hand-Crafted Feature and Shallow Predictor
#### 4.1 Geometric Feature with Shallow Predictor
#### 4.2 Appearance Feature with Shallow Predictor

Week12进度就汇报到这里为止，剩下的放到final👋🏻

---
### 5. FBP Evaluation via Deep Predictor
三种CNN模型
- AlexNet <= 227*227 random crop of raw image
- ResNet- 18 <= 224*224 random crop of raw image
- ResNeXt-50 <= 224*224 random crop of raw image

>All these CNN models are trained by initializing weights using networks pre-trained on the ImageNet dataset. 

Two different experiment settings as following
- 5-fold cross validation
- 60% for train, 40% for test

### 6. Conclusion

## References
[机器学习](https://zh.wikipedia.org/wiki/机器学习) 

[人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络) 

[卷积神经网络](https://zh.wikipedia.org/wiki/卷积神经网络)

[交叉验证](https://zh.wikipedia.org/wiki/交叉验证)

[过拟合](https://zh.wikipedia.org/wiki/过拟合)

[AlexNet](https://my.oschina.net/u/876354/blog/1633143)

[PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)