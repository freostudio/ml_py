##knn-

#%%
##K邻居分类算法
# 案例1-分析乳腺癌数据集合
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
#%%
##Step1-训练数据与测试数据
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
%matplotlib inline 
import matplotlib.pyplot as plt

X_train,X_test,y_train,y_test = train_test_split(cancer['data'],cancer['target'],stratify= cancer['target'],random_state=66)

##Step2-预测精度
training_accuracy =[]
test_accuracy =[]

#k值-从1-11
n_neighbors_setting = range(1,11)

for k in n_neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))##训练集合精度
    test_accuracy.append(clf.score(X_test,y_test))##泛化精度

##Step3-预测结果精度展示-选择最佳的K值
plt.plot(n_neighbors_setting,training_accuracy,marker='o', label='training_accuracy')
plt.plot(n_neighbors_setting,test_accuracy,marker='s', label='test_accuracy')
plt.ylabel("Accuracy")
plt.xlabel("knn-k-neighbors")
plt.legend()


# %%
##K邻居回归算法
# 案例2-波士顿房价
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
#案例3 
# import mglearn
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
##案例4-
from  sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np 
import mglearn
%matplotlib inline 
import matplotlib.pyplot as plt

wave_datasets = mglearn.datasets.make_wave(n_samples=40)
#rint(wave_datasets)
#Step1-将wave数据进行分割训练集和测试集合
X_train,X_test,y_train,y_test = train_test_split(wave_datasets[0],wave_datasets[1],random_state=0)
##Step2-模型化，k=3
reg  = KNeighborsRegressor(n_neighbors=3)
#Step3-训练拟合模型
reg.fit(X_train,y_train)
#Step4-测试值
reg.predict(X_test)
#Step5-评估模型
reg.score(X_test,y_test)

#Step6-分析不同K值的模型效果
n_neighbors_setting =range(1,5)
training_accuracy =[]
test_accuracy = []

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax  in  zip(n_neighbors_setting, axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    training_accuracy.append(reg.score(X_train,y_train))##训练集合精度
    test_accuracy.append(reg.score(X_test,y_test))##泛化精度
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'o',c=mglearn.cm2(1),markersize=8)
    ax.plot(line,reg.predict(line))
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    
axes[0].legend(["Model predictions", "Training data/target",
            "Test data/target"], loc="best")


 




# %%
##Step7-预测结果精度展示-选择最佳的K值
plt.plot(n_neighbors_setting,training_accuracy,marker='o', label='training_accuracy')
plt.plot(n_neighbors_setting,test_accuracy,marker='s', label='test_accuracy')
plt.ylabel("Accuracy")
plt.xlabel("knn-k-neighbors")
plt.legend()
# %%
##KNN用于线性模型
##1-Logistics 回归
##2-Linear Support vector machine 线性SVM 支持向量机
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np 
import matplotlib.pyplot as plt


X,y= mglearn.datasets.make_forge()
fig,axes = plt.subplot(1,2,figsize=(10,3))


for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()




# %%

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn
##Step1 数据准备
X,y = make_blobs(random_state=42)
#X为2维数组,矩阵 A100x2，y 一维度 矩阵 B100x1
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("f0")
plt.xlabel("f1")
plt.legend([['class 0','class 1','class 2']])
plt.show()

##Step2 



# %%
