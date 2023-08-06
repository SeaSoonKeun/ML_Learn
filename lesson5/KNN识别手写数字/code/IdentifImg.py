# coding:utf-8
'''
Created on 2020年1月11日
@author: zfg
'''

import os
from KNNDateOnHand import *

# 将每一个文件的内容拼接成一行
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        if (lineStr != "\n"):
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def IdentifImgClassTest():
    # 分类号（数字）
    hwLabels = []
    # 获取TrainData目录下所有的文件
    # os 模块提供了非常丰富的方法用来处理文件和目录
    # os.listdir(path) 返回path指定的文件夹包含的文件或文件夹的名字的列表
    trainingFileList = os.listdir('../data/TrainData')
    # 训练集的数据量，空间中分布的数据量  m=1934, 1934个文件
    m = len(trainingFileList)
    # 1934*1024的零矩阵
    # 构建一个m行1024列的零矩阵，用于存储所有的训练数据
    # 为什么是1024 列，因为每个文件中有32行，每行有32个字符，所以每个文件有1024个字符
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 以'.'为分隔符，将文件名分割为两部分，取第一部分
        fileStr = fileNameStr.split('.')[0]
        # 以'_'为分隔符，将文件名分割为两部分，取第一部分
        classNumStr = int(fileStr.split('_')[0])
        # 将分类号添加到hwLabels中
        hwLabels.append(classNumStr)
        # 将训练集数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('../data/TrainData/%s' % fileNameStr)

    testFileList = os.listdir('../data/TestData')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('../data/TestData/%s' % fileNameStr)
        # 将训练集数据和第一条测试数据 以及K值传递给classify这个方法
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("识别出的数字是: %d, 真实数字是: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n识别错误次数 %d" % errorCount)
    errorRate = errorCount / float(mTest)
    print("\n正确率: %f" % (1 - errorRate))


if __name__ == '__main__':
    IdentifImgClassTest()
