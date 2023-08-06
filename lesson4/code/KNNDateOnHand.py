# coding:utf-8
'''
Created on 2020年1月11日
@author: zfg
'''

import numpy as np
import operator
import matplotlib.pyplot as plt
from array import array
from matplotlib.font_manager import FontProperties


def classify(normData, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # shape[0]代表的是矩阵的行数
    # tile的英文意思是瓷砖，tile(A, reps):构造一个矩阵，通过A重复reps次得到
    diffMat = np.tile(normData, (dataSetSize, 1)) - dataSet # tile函数的作用是将normData这个向量复制成dataSetSize行1列的矩阵，然后再与dataSet相减，x1-x2 为了求欧式距离
    sqDiffMat = diffMat ** 2 # 平方
    sqDistances = sqDiffMat.sum(axis=1) # axis=1代表的是按照列相加，axis=0代表的是按照行相加
    distance = sqDistances ** 0.5 # 开根号
    sortedDistIndicies = distance.argsort() # 对距离排序，argsort函数返回的是数组值从小到大的索引值
    #     classCount保存的K是魅力类型   V:在K个近邻中某一个类型的次数
    classCount = {} # {}代表的是字典
    for i in range(k): # 选择距离最小的K个点
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # classCount.get(voteLabel, 0) 从classCount中获取voteLabel对应的值，如果不存在，则返回0
        '''
        sorted 是 python 内置的排序函数，sorted(iterable, cmp=None, key=None, reverse=False)，返回一个新的列表
        classCount.items() 返回的是一个列表，列表中的元素是元组，元组的第一个元素是key，第二个元素是value
        key=operator.itemgetter(1) 代表的是获取元组的第二个元素的值，也就是按照value进行排序
        reverse=True 代表的是降序排列
        '''
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # 返回出现次数最多的那个类型


def file2matrix(filename):
    fr = open(filename)
    #     readlines:是一次性将这个文本的内容全部加载到内存中(列表)
    arrayOflines = fr.readlines()
    numOfLines = len(arrayOflines)
    #    zeros:生成一个numOfLines行3列的矩阵，矩阵中的值都是0
    returnMat = np.zeros((numOfLines, 3))
    #    存储标签
    classLabelVector = []
    index = 0
    #    遍历每一行
    for line in arrayOflines:
        #        strip:去掉每一行的回车
        line = line.strip()
        #        split:按照tab键进行切割
        # print(line.split('\t'))
        #       将每一行的数据存储到列表中
        listFromline = list(map(float, line.split('\t')))
        #       将每一行的数据的前三列存储到returnMat中，也就是特征值
        returnMat[index, :] = listFromline[0:3]
        #       将每一行的数据的最后一列存储到classLabelVector中，也就是标签
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector


'''
    将训练集中的数据进行归一化
    归一化的目的：
        训练集中飞行公里数这一维度中的值是非常大，那么这个纬度值对于最终的计算结果(两点的距离)影响是非常大，
        远远超过其他的两个维度对于最终结果的影响
    实际约会姑娘认为这三个特征是同等重要的
    下面使用最大最小值归一化的方式将训练集中的数据进行归一化
'''


# 归一化：把所有数据都缩小到0-1之间
# 为什么需要归一化，因为不同的特征值的取值范围不一样，比如说飞行公里数的取值范围是0-100000，而玩游戏的时间的取值范围是0-10
# 这样的话，飞行公里数这个特征值对于最终的计算结果(两点的距离)影响是非常大，远远超过其他的两个维度对于最终结果的影响
# 实际约会姑娘认为这三个特征是同等重要的
def autoNorm(dataSet):
    #     dataSet.min(0)   代表的是统计这个矩阵中每一列的最小值     返回值是一个矩阵1*3矩阵
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0] # m代表的是矩阵的行数
    #     normDataSet存储归一化后的数据 np.shape(dataSet)返回的是矩阵的行数和列数 np.zeros()生成一个和dataSet一样大小的矩阵，但是里面的值都是0
    normDataSet = np.zeros(np.shape(dataSet)) # 生成一个和dataSet一样大小的矩阵，但是里面的值都是0
    # np.tile(minVals, (m, 1))  tile函数的作用是将minVals这个向量复制成m行1列的矩阵
    # dataSet - np.tile(minVals, (m, 1))  将dataSet这个矩阵中的每一列都减去minVals这个向量
    normDataSet = dataSet - np.tile(minVals, (m, 1)) # tile函数的作用是将minVals这个向量复制成m行1列的矩阵，然后再与dataSet相减，实现归一化
    # np.tile(ranges, (m, 1))  tile函数的作用是将ranges这个向量复制成m行1列的矩阵, ranges代表的是每一列的最大值和最小值的差值
    # normDataSet / np.tile(ranges, (m, 1))  将dataSet这个矩阵中的每一列都除以ranges这个向量
    # 归一化公式：newValue = (oldValue - min) / (max - min)
    normDataSet = normDataSet / np.tile(ranges, (m, 1)) # 特征值相除
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1 # 测试集占总数据的比例
    datingDataMat, datingLabels = file2matrix('../data/datingTestSet2.txt')

    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)

    #     shape获取矩阵的行数以及列数，以二元组的形式返回的
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio) # 测试集的数量
    errorCount = 0.0 # 错误率
    for i in range(numTestVecs):
        # normMat[i, :]代表的是测试集中的第i个样本的所有列的值，normMat[numTestVecs:m, :]切片操作选择了normMat数组的第numTestVecs行到第m-1行的所有列的数据
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], \
                                    datingLabels[numTestVecs:m], 4) # datingLabels[numTestVecs:m]切片操作选择了datingLabels数组的第numTestVecs个元素到第m-1个元素的数据
        print('模型预测值: %d ,真实值 : %d' \
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    errorRate = errorCount / float(numTestVecs)
    print('正确率 : %f' % (1 - errorRate))
    return 1 - errorRate


'''
    拿到每条样本的飞行里程数和玩视频游戏所消耗的事件百分比这两个维度的值，使用散点图
'''

# 创建散点图 scatter英文含义是分散，比如词组 scatter the seeds 撒种子
def createScatterDiagram():
    datingDataMat, datingLabels = file2matrix('../data/datingTestSet2.txt')
    type1_x = [] # [] 代表的是列表,列表中的元素可以是任意类型, type1 代表的是不喜欢
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    fig = plt.figure() # 创建一个新的图像
    axes = plt.subplot(111) # 111代表的是将画布分割成1行1列，图像画在从左到右从上到下的第1块
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    for i in range(len(datingLabels)): # len(datingLabels)代表的是datingLabels这个列表的长度
        if datingLabels[i] == 1:  # 不喜欢
            type1_x.append(datingDataMat[i][0]) # datingDataMat[i][0]代表的是第i个样本的飞行里程数
            type1_y.append(datingDataMat[i][1]) # datingDataMat[i][1]代表的是第i个样本的玩视频游戏所消耗的事件百分比

        if datingLabels[i] == 2:  # 魅力一般
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:  # 极具魅力
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=50, c='blue')
    plt.xlabel(u'每年飞行里程数')
    plt.ylabel(u'玩视频游戏所消耗的事件百分比')
    # axes.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)
    plt.scatter(datingDataMat[:, 0], datingDataMat[:, 1], c=datingLabels) # 画散点图,第一个参数是x轴的值,第二个参数是y轴的值,第三个参数是点的颜色
    plt.show()


def classifyperson():
    resultList = ['没感觉', '看起来还行', '极具魅力']
    input_man = [0, 0, 0]
    datingDataMat, datingLabels = file2matrix('../data/datingTestSet2.txt') # 读取数据,返回数据矩阵和标签向量
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化,返回归一化后的矩阵,极值差,最小值
    result = classify((input_man - minVals) / ranges, normMat, datingLabels, 10) # 分类,返回分类结果,输入数据归一化后的数据,训练集,标签,最近邻数目,默认为10
    print('你即将约会的人是:', resultList[result - 1])


if __name__ == '__main__':
    #     createScatterDiagram观察数据的分布情况
    createScatterDiagram()
    acc = datingClassTest()
    # print('正确率 : %f' % (acc))
    if (acc > 0.9):
        classifyperson()
