package com.msb.lr_new

import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession


object LinearRegression02 {
  //0.5395881831013476
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")

    val spark = SparkSession.builder().config(conf).appName("LinearRegression").getOrCreate() //创建环境变量实例

    var data = spark.read.format("libsvm") //读取libsvm格式的数据
      .load("data/sample_linear_regression_data.txt") //加载数据

   //增加一列与第一个特征一模一样 -0.294192922737251,-0.294192922737251
   //未增加：   -0.5883852628595317

    // w代表影响因子，越大越重要

    /**
      *randomSplit 是随机切割的方法  Array(0.8,0.2)
      *
      *  data拆分成两个df
      *  第一个df的数据量是data  的  80%   随机
      *  第一个df的数据量是data  的  20%
      *
      *  seed种子如果每次都是一样的
      *  那么每次随机出来的数据都是一样的
      *
      *  测试环境为什么随机出来的数据一样？
      *  便于排查错误
      *
      *
      *  为什么要把data切成两份？
      *  1、第一份做训练集   --   model
      *  2、第二份做测试集
      */

    val DFS = data.randomSplit(Array(0.8,0.2),1) //1代表随机种子, 0.8代表训练集占比,0.2代表测试集占比, dfs 代表dataframe的数组

    val (training,test) = (DFS(0),DFS(1)) //训练集和测试集

    val lr = new LinearRegression()
      .setMaxIter(10) //最大迭代次数
      //L1+L2系数之和    0代表不使用正则化
//      .setRegParam(0.3)
    /**
      * 用于调整L1、L2之间的比例，简单说:调整L1，L2前面的系数
      * For alpha = 0, the penalty is an L2 penalty.
      * For alpha = 1, it is an L1 penalty.
      * For alpha in (0,1), the penalty is a combination of L1 and L2.
      */
//      .setElasticNetParam(0.8)


    // Fit the model
    val lrModel = lr.fit(training) //训练模型

    // 打印模型参数w1...wn和截距w0
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // 基于训练集数据，总结模型信息
    val trainingSummary = lrModel.summary
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")

    spark.close()

  }
}
