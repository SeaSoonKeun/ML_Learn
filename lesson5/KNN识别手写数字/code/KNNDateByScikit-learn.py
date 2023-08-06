# coding:utf-8
'''
Created on 2020年1月11日

@author: root
'''
from sklearn.neighbors._unsupervised import NearestNeighbors
import numpy as np
from KNNDateOnHand import *

datingDataMat, datingLabels = file2matrix('../../../lesson4/data/datingTestSet2.txt')
normMat, ranges, minVals = autoNorm(datingDataMat)

# 1.使用sklearn中的KNN算法, NearestNeighbors 最近邻算法, n_neighbors=10, 代表的是取最近的10个点, 默认是5个点, fit()方法是归一化后的训练模型
nbrs = NearestNeighbors(n_neighbors=10).fit(normMat)
input_man = [[50000, 8, 9.5]]
# 2.使用kneighbors()方法, 返回的是距离和索引, distances是距离, indices是索引
S = (input_man - minVals) / ranges
distances, indices = nbrs.kneighbors(S)
# classCount   K：类别名    V：这个类别中的样本出现的次数

classCount = {}
# 从最近的10个点中, 找出出现次数最多的类别
for i in range(10):
    voteLabel = datingLabels[indices[0][i]]
    classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
resultList = ['没感觉', '看起来还行', '极具魅力']
print(resultList[sortedClassCount[0][0] - 1])
