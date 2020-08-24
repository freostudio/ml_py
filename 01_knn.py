##knn
#%%
##案例1-分析乳腺癌数据集合
##https://www.programcreek.com/python/example/104690/sklearn.datasets.load_breast_cancer
##数据集合中：569个数据点（行），30个维度特征（列）组成的矩阵
##1-其中：212个体数据target标志分类为恶性-1，357个数据target标准分类为良性-0

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['data'].shape)
print(cancer['data'][:2])##各个维度（特征值）：0-1
print(cancer['target'].shape)
print(cancer['target'][:10])##分类标签
print(cancer['target_names'].shape)
print(cancer['target_names'])##分类标签名
print(cancer['DESCR'])##描述
print(cancer['feature_names'])##每一列的维度/特征名称
print(cancer['filename'])##数据集合csv文件名称


# %%
##案例2-波士顿房价
##https://www.programcreek.com/python/example/82897/sklearn.datasets.load_boston
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())
print(boston['data'].shape)
print(boston['data'][:2])##各个维度（特征值）：0-1
print(boston['target'].shape)
print(boston['target'][:10])##分类标签
print(boston['DESCR'])##描述
print(boston['feature_names'])##每一列的维度/特征名称
print(boston['filename'])##数据集合csv文件名称


# %%
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np 
data= mglearn.datasets.make_forge()
#分割数据
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
#设置K值
clf = KNeighborsClassifier(n_neighbors=3)
#fit
clf.fit(X_train,y_train)
#评估模型
#y_predict = clf.predict(X_test)
#np.mean(y_predict==y_test)
clf.score(X_test,y_test)
# %%
