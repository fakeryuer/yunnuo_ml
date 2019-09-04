#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 本地数据集
data = pd.read_csv('./iris.csv')
data.head()


# 对文本值进行数值化处理
mapping_dict= {
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
}
data['class']=data['class'].map(mapping_dict)

# 1 取出数据当中的特征值和目标值
x = data.drop(['class'], axis=1)
y = data['class']

# 2 数据分割成训练集和测试集 test_size=0.2表示将20%的数据用作测试集r
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 3 数据标准化
std = StandardScaler()

std.fit_transform(x_train)
std.transform(x_test)

# 4 建立KNN算法，初始化参数
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train) # 将测试集送入算法
y_predict = knn.predict(x_test) # 获取预测结果


# numpy dataframe 转换
list_test = np.array(y_test)


print("准确率：",knn.score(x_test, y_test))
# 查看每次预测
labels = ["setosa", "versicolor", "virginica"]
for i in range(len(y_predict)):
    print("第%d次测试:真实值:%s\t\t预测值:%s\t"%((i+1), labels[y_predict[i]], labels[list_test[i]]))
    print(("是否正确：%s"%('True' if (y_predict[i] == list_test[i]) else '*** False ***')))



