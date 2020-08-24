#%%
##1-线性回归-最小二乘法
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn

X,y =mglearn.datasets.make_wave()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

lr = LinearRegression().fit(X_train,y_train)
print("线性回归模型的斜率w：{}".format(lr.coef_))
print("线性回归模型的截距b：{}".format(lr.intercept_))
print("训练集模型评估得分：{}".format(lr.score(X_train,y_train)))
print("测试集模型评估得分：{}".format(lr.score(X_test,y_test)))



# %%
##2-岭回归
from sklearn.linear_model import Ridge
##alpha系数默认1
ridge = Ridge().fit(X_train,y_train)
print("训练集模型评估得分：{}".format(ridge.score(X_train,y_train)))
print("测试集模型评估得分：{}".format(ridge.score(X_test,y_test)))



# %%
##alpha=10时候
ridge = Ridge(alpha=0.1).fit(X_train,y_train)
print("训练集模型评估得分：{}".format(ridge.score(X_train,y_train)))
print("测试集模型评估得分：{}".format(ridge.score(X_test,y_test)))

# %%
##3-Lasso 线性回归
#约束系数使其变为0