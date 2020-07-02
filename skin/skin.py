#-*- coding:utf-8 -*-
# import utils
import pyod
import os
import time
import csv
import scipy
import math
import random
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import orangecontrib.associate.fpgrowth as oaf
from apyori import apriori
from array import array
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from pyod.utils.utility import precision_n_scores
# import models
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
# import pyod functions
from pyod.utils.data import generate_data, evaluate_print, get_outliers_inliers
from pyod.utils.example import visualize


# 读取 diff 文件
with open("/root/dataMining/lastweek/data/skin/meta_data/skin.diff.csv","r",encoding='utf-8-sig') as fulldata:
    reader = csv.reader(fulldata)
    lines = list(reader)
print("This dataset has {} items.".format(len(lines)-1))
print("The index/attributes of this dataset are:")
index = lines[0]
for i in range(len(lines[0])):
    print("No.{}\t {}".format(i+1, lines[0][i]))

lines = lines[1:]

ItemNo = [x+1 for x in range(len(lines))]
df = pd.DataFrame(lines, index=ItemNo, columns=index)
df.describe()

pd_fulldata = pd.read_csv("/root/dataMining/lastweek/data/skin/meta_data/skin.diff.csv")
pd_fulldata.head(10)

# 打乱 diff 顺序
np.random.seed(0)
pd_fulldata = pd_fulldata.reindex(np.random.permutation(pd_fulldata.index))
pd_fulldata.head(10)

# 组织特征和标签信息
fulldata = pd_fulldata[["R", "G", "B"]]
fulldata = fulldata.values.tolist()

fulllabel = pd_fulldata[["ground.truth"]]
fulllabel = fulllabel.values.tolist()

# reformat label info
for i in range(len(fulllabel)):
    if fulllabel[i][0] == "nominal":
        fulllabel[i] = 0
    else:
        fulllabel[i] = 1


# 将 diff 的训练集和测试集分开
indexInterval = 0.80
indexCutter = int(len(fulldata) * indexInterval)
X_train = np.array(fulldata[0:indexCutter])
Y_train = np.array(fulllabel[0:indexCutter])
X_test = np.array(fulldata[indexCutter:])
Y_test = np.array(fulllabel[indexCutter:])
print("TRAIN SET LEN:", len(X_train), "TEST SET LEN:", len(X_test))

x_outliers, x_inliers = get_outliers_inliers(X_train, Y_train)
train_inliers = len(x_inliers)
train_outliers = len(x_outliers)
print('TRAIN SET OUTLIERS : ',train_outliers,'TRAIN SET INLIERS : ',train_inliers)


# 准备输出文件
outputfile = open("skindiffResults.txt", "w")
print("-*- fulldata -*-", file=outputfile)


# 获取 roc 和 prn 参数
def evaluate(y, y_pred):
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)
    roc = np.round(roc_auc_score(y, y_pred), decimals=4)
    prn = np.round(precision_n_scores(y, y_pred), decimals=4)
    return roc, prn


# 准备diff文件的离群点分类器
random_state = np.random.RandomState(42)
#outliers_fraction = 0.05
diffKNN = KNN()
diffKNNmean = KNN(method='mean')
diffPCA = PCA()
diffMCD = MCD()
diffCBLOF = CBLOF(check_estimator=False, random_state=random_state)
diffFB = FeatureBagging(LOF(n_neighbors=35),check_estimator=False,random_state=random_state)
diffHBOS = HBOS()
diffIF = IForest(random_state=random_state)

# begin to train full data classifiers
diffClassifiers = [diffKNN, diffKNNmean, diffPCA, diffMCD, diffCBLOF, diffFB, diffHBOS, diffIF]
diffClassifiersName = ['K Nearest Neighbors (KNN)', 'Average KNN (avg KNN)','Principal Component Analysis (PCA)','Minimum Covariance Determinant (MCD)','Cluster-based Local Outlier Factor (CBLOF)','Feature Bagging','Histogram-base Outlier Detection (HBOS)','Isolation Forest']
for clf, clf_name in zip(diffClassifiers,diffClassifiersName):
    clf.fit(X_train)
    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_  

    # get the prediction on the test data
    X_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    X_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, Y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, Y_test, X_test_scores)

    # calculate roc and prn, print out to file
    print("===== clf name 【{}】".format(clf_name), file=outputfile)
    print("===== train", file=outputfile)
    roc, prn = evaluate(Y_train, y_train_scores)
    print("roc:{} prn:{}".format(roc,prn), file=outputfile)
    roc, prn = evaluate(Y_test, X_test_scores)
    print("===== test", file=outputfile)
    print("roc:{} prn:{}\n".format(roc,prn),file=outputfile)
    print("====================================================================")

# 关闭 diff 分类器信息输出文件
outputfile.close()

# benchmark 离群点检测分类器设定
classifiers = {
    'K Nearest Neighbors (KNN)': KNN(),
    'Average KNN (avg KNN)': KNN(method='mean'),
    'Principal Component Analysis (PCA)': PCA(),
    'Minimum Covariance Determinant (MCD)': MCD(),
    'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(check_estimator=False, random_state=random_state),
    'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),check_estimator=False,random_state=random_state),
    'Histogram-base Outlier Detection (HBOS)': HBOS(),
    'Isolation Forest': IForest(random_state=random_state)
}

# 准备开始处理各个 benchmark 文件
benchmarkFileBaseName = "/root/dataMining/lastweek/data/skin/benchmarks/skin_benchmark_"
benchmarkFileNumber = 1740
benchmarkCurrenFile = 1
timebar = tqdm(range(benchmarkFileNumber-benchmarkCurrenFile))

# 准备输出文件
outputfile = open("skinBenchmarkResults.txt", "w")
print("-*- benchmark -*-", file=outputfile)
jumpedFile = 0

for item in timebar:
    benchmarkNumber = "%04d" % benchmarkCurrenFile
    fileName = benchmarkFileBaseName + benchmarkNumber + ".csv"
    print("###################################################################################", file=outputfile)
    print("##### "+fileName+" #####", file=outputfile)
    print("###################################################################################\n", file=outputfile)
    
    benchmarkCurrenFile += 1
    # 判断文件是否存在
    if not os.path.exists(fileName):
        continue

    # 读取 benchmark 信息
    benchmark_data = pd.read_csv(fileName)
    # benchmark 文件本身就是乱序的，因此可直接抽样，不用再打乱
    benchmarkdata = benchmark_data[["R", "G", "B"]]
    benchmarkdata = benchmarkdata.values.tolist()

    benchmarklabel = benchmark_data[["ground.truth"]]
    benchmarklabel = benchmarklabel.values.tolist()

    # reformat label info
    for i in range(len(benchmarklabel)):
        if benchmarklabel[i][0] == "nominal":
            benchmarklabel[i] = 0
        else:
            benchmarklabel[i] = 1

    # benchmark 的训练集和测试集分开
    indexCutter = int(len(benchmark_data) * indexInterval)
    benchmark_X_train = np.array(benchmarkdata[:indexCutter])
    benchmark_Y_train = np.array(benchmarklabel[:indexCutter])
    benchmark_X_test = np.array(benchmarkdata[indexCutter:])
    benchmark_Y_test = np.array(benchmarklabel[indexCutter:])

    # 判断该 benchmark 是否有两种 label，没有则跳过该 benchmark
    flag_train = False
    flag_test = False
    if (sum(benchmark_Y_train == 0) and sum(benchmark_Y_train == 1)):
        flag_train = True
    if (sum(benchmark_Y_test == 0) and sum(benchmark_Y_test == 1)):
        flag_test = True
    if not (flag_train and flag_test):
        jumpedFile += 1
        continue

    timebar.set_description(f'wine_benchmark_{benchmarkNumber} TR-SIZE:{len(benchmark_X_train)} TE-SIZE:{len(benchmark_X_test)} SKIPED:{jumpedFile}')
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # prepare diff classifier
        diffCLF = diffClassifiers[i]

        # train benchmark's classifier
        clf.fit(benchmark_X_train)
        benchmark_y_train_pred = clf.labels_
        benchmark_y_train_scores = clf.decision_scores_

        # get the prediction on the test data
        diff_test_pred = diffCLF.predict(benchmark_X_test)
        diff_test_scores = diffCLF.decision_function(benchmark_X_test)
        benchmark_test_pred = clf.predict(benchmark_X_test)  # outlier labels (0 or 1)
        benchmark_test_scores = clf.decision_function(benchmark_X_test)  # outlier scores

        # print CLF name
        print("CLF: 【{}】".format(clf_name), file=outputfile)

        # evaluate and print the results
        print("On Training Data:", file=outputfile)
        roc, prn = evaluate(benchmark_Y_train, benchmark_y_train_scores)
        print("roc:{} - prn:{}".format(roc, prn), file=outputfile)
        
        print("\nOn Test Data:", file=outputfile)
        roc, prn = evaluate(benchmark_Y_test, benchmark_test_scores)
        print("roc:{} - prn:{}".format(roc, prn), file=outputfile)

        print("\nOn Diff CLassifier:", file=outputfile)
        roc, prn = evaluate(benchmark_Y_test, diff_test_scores)
        print("roc:{} - prn:{}".format(roc, prn), file=outputfile)
        print("\n===================================================================================\n", file=outputfile)

outputfile.close()